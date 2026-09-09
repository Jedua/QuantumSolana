#include "orderbook.hpp"
#include "shm_writer.hpp"

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include "simdjson.h"

#include <iostream>
#include <string>
#include <string_view>
#include <memory>
#include <cstdlib>
#include <chrono>

namespace beast = boost::beast;         
namespace http = beast::http;           
namespace websocket = beast::websocket; 
namespace net = boost::asio;            
namespace ssl = boost::asio::ssl;       
using tcp = boost::asio::ip::tcp;       

namespace quantsolana {

class BinanceFeed : public std::enable_shared_from_this<BinanceFeed> {
    tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    beast::flat_buffer buffer_;
    std::string host_;
    std::string port_;
    std::string target_;
    
    // Instantiated outside the loop to reuse memory buffer
    simdjson::dom::parser parser_;
    OrderbookState ob_state_{};
    uint64_t update_counter_{0};
    double prev_best_bid_price_{0.0};
    double prev_best_bid_qty_{0.0};
    double prev_best_ask_price_{0.0};
    double prev_best_ask_qty_{0.0};

    net::steady_timer reconnect_timer_;

public:
    explicit BinanceFeed(net::io_context& ioc, ssl::context& ctx)
        : resolver_(net::make_strand(ioc))
        , ws_(net::make_strand(ioc), ctx)
        , reconnect_timer_(ioc)
        , host_("fstream.binance.com")
        , port_("443")
        , target_("/ws/solusdt@depth10@100ms")
    {
    }

    void run() {
        do_resolve();
    }

    const OrderbookState& get_state() const {
        return ob_state_;
    }

private:
    void do_resolve() {
        resolver_.async_resolve(
            host_,
            port_,
            beast::bind_front_handler(&BinanceFeed::on_resolve, shared_from_this()));
    }

    void on_resolve(beast::error_code ec, tcp::resolver::results_type results) {
        if (ec) {
            fail(ec, "resolve");
            return reconnect();
        }

        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));

        beast::get_lowest_layer(ws_).async_connect(
            results,
            beast::bind_front_handler(&BinanceFeed::on_connect, shared_from_this()));
    }

    void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type ep) {
        if (ec) {
            fail(ec, "connect");
            return reconnect();
        }

        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));

        if(! SSL_set_tlsext_host_name(ws_.next_layer().native_handle(), host_.c_str())) {
            ec = {static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
            fail(ec, "ssl_set_tlsext_host_name");
            return reconnect();
        }

        ws_.next_layer().async_handshake(
            ssl::stream_base::client,
            beast::bind_front_handler(&BinanceFeed::on_ssl_handshake, shared_from_this()));
    }

    void on_ssl_handshake(beast::error_code ec) {
        if (ec) {
            fail(ec, "ssl_handshake");
            return reconnect();
        }

        beast::get_lowest_layer(ws_).expires_never();
        
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
            }));

        // Boost.Beast auto ping/pong handling. We can also hook custom logic here if needed.
        ws_.control_callback(
            [this](websocket::frame_type kind, beast::string_view payload) {
                // Keep-alive management if we want custom tracking
            });

        ws_.async_handshake(host_, target_,
            beast::bind_front_handler(&BinanceFeed::on_handshake, shared_from_this()));
    }

    void on_handshake(beast::error_code ec) {
        if (ec) {
            fail(ec, "handshake");
            return reconnect();
        }

        do_read();
    }

    void do_read() {
        ws_.async_read(
            buffer_,
            beast::bind_front_handler(&BinanceFeed::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec) {
            fail(ec, "read");
            return reconnect();
        }

        process_message();

        // Clear buffer before next read
        buffer_.consume(buffer_.size());

        do_read(); // Continue read loop
    }
    
    // Fast string to double conversion avoiding locale overhead
    double fast_atof(std::string_view str) {
        return std::strtod(str.data(), nullptr);
    }

    void process_message() {
        try {
            // Fast DOM JSON parsing
            simdjson::dom::element doc = parser_.parse(
                reinterpret_cast<const char*>(buffer_.data().data()),
                buffer_.size()
            );

            simdjson::dom::array bids;
            if (doc["bids"].get_array().get(bids) != simdjson::SUCCESS) {
                if (doc["b"].get_array().get(bids) != simdjson::SUCCESS) {
                    return;
                }
            }

            simdjson::dom::array asks;
            if (doc["asks"].get_array().get(asks) != simdjson::SUCCESS) {
                if (doc["a"].get_array().get(asks) != simdjson::SUCCESS) {
                    return;
                }
            }

            // Bids extraction
            size_t i = 0;
            for (simdjson::dom::array pair : bids) {
                if (i >= 10) break;
                std::string_view price_str = pair.at(0).get_string().value();
                std::string_view qty_str = pair.at(1).get_string().value();
                
                ob_state_.bids[i].price = fast_atof(price_str);
                ob_state_.bids[i].qty = fast_atof(qty_str);
                ++i;
            }

            // Asks extraction
            i = 0;
            for (simdjson::dom::array pair : asks) {
                if (i >= 10) break;
                std::string_view price_str = pair.at(0).get_string().value();
                std::string_view qty_str = pair.at(1).get_string().value();
                
                ob_state_.asks[i].price = fast_atof(price_str);
                ob_state_.asks[i].qty = fast_atof(qty_str);
                ++i;
            }

            // Extraer o generar update ID y timestamp ms
            uint64_t update_id = 0;
            if (doc["u"].get_uint64().get(update_id) == simdjson::SUCCESS || doc["lastUpdateId"].get_uint64().get(update_id) == simdjson::SUCCESS) {
                ob_state_.last_update_id = update_id;
            } else {
                ob_state_.last_update_id = ++update_counter_;
            }

            uint64_t ts_ms = 0;
            if (doc["E"].get_uint64().get(ts_ms) == simdjson::SUCCESS || doc["T"].get_uint64().get(ts_ms) == simdjson::SUCCESS) {
                ob_state_.timestamp_ms = ts_ms;
            } else {
                ob_state_.timestamp_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count()
                );
            }

            // Cálculo inmediato de métricas de microestructura de mercado
            double cur_bid_price = ob_state_.bids[0].price;
            double cur_bid_qty = ob_state_.bids[0].qty;
            double cur_ask_price = ob_state_.asks[0].price;
            double cur_ask_qty = ob_state_.asks[0].qty;

            if (cur_bid_price > 0.0 && cur_ask_price > 0.0) {
                ob_state_.mid_price = (cur_bid_price + cur_ask_price) / 2.0;
                ob_state_.spread = cur_ask_price - cur_bid_price;

                // Cálculo de Imbalance acumulado (Top 10 niveles)
                double total_bid_qty = 0.0;
                double total_ask_qty = 0.0;
                for (size_t k = 0; k < 10; ++k) {
                    total_bid_qty += ob_state_.bids[k].qty;
                    total_ask_qty += ob_state_.asks[k].qty;
                }
                if (total_bid_qty + total_ask_qty > 0.0) {
                    ob_state_.imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty);
                } else {
                    ob_state_.imbalance = 0.0;
                }

                // Cálculo de Order Flow Imbalance (OFI) y EMA5
                double e_b = 0.0;
                if (prev_best_bid_price_ > 0.0) {
                    if (cur_bid_price > prev_best_bid_price_) {
                        e_b = cur_bid_qty;
                    } else if (cur_bid_price == prev_best_bid_price_) {
                        e_b = cur_bid_qty - prev_best_bid_qty_;
                    } else {
                        e_b = -prev_best_bid_qty_;
                    }
                }

                double e_a = 0.0;
                if (prev_best_ask_price_ > 0.0) {
                    if (cur_ask_price < prev_best_ask_price_) {
                        e_a = cur_ask_qty;
                    } else if (cur_ask_price == prev_best_ask_price_) {
                        e_a = cur_ask_qty - prev_best_ask_qty_;
                    } else {
                        e_a = -prev_best_ask_qty_;
                    }
                }

                ob_state_.ofi_t = e_b - e_a;
                double alpha_5 = 2.0 / (5.0 + 1.0);
                ob_state_.ofi_ema5 = alpha_5 * ob_state_.ofi_t + (1.0 - alpha_5) * ob_state_.ofi_ema5;

                prev_best_bid_price_ = cur_bid_price;
                prev_best_bid_qty_ = cur_bid_qty;
                prev_best_ask_price_ = cur_ask_price;
                prev_best_ask_qty_ = cur_ask_qty;
            }

            // Write to shared memory on every valid depth update
            if (shm_writer_) {
                shm_writer_->write_state(ob_state_);
            }

        } catch (const simdjson::simdjson_error& e) {
            // Fallback for parser exception, non-fatal to the connection
            std::cerr << "JSON Parsing error: " << e.what() << "\n";
        }
    }

    void fail(beast::error_code ec, char const* what) {
        std::cerr << "Network Error - " << what << ": " << ec.message() << "\n";
    }
    
    void reconnect() {
        std::cerr << "Attempting to reconnect in 3 seconds...\n";
        reconnect_timer_.expires_after(std::chrono::seconds(3));
        reconnect_timer_.async_wait([this](beast::error_code ec) {
            if (!ec) {
                do_resolve();
            }
        });
    }

public:
    void set_shm_writer(std::shared_ptr<SharedMemoryWriter> writer) {
        shm_writer_ = writer;
    }

private:
    std::shared_ptr<SharedMemoryWriter> shm_writer_;
};

} // namespace quantsolana

int main(int argc, char* argv[]) {
    try {
        std::cout << "[BINANCE INGESTION V10] Iniciando ejecutable real C++ Binance WebSocket...\n";
        
        auto shm_writer = std::make_shared<quantsolana::SharedMemoryWriter>("QuantSolana_V10_SHM");
        
        net::io_context ioc;
        ssl::context ctx{ssl::context::tlsv12_client};
        ctx.set_default_verify_paths();

        auto feed = std::make_shared<quantsolana::BinanceFeed>(ioc, ctx);
        feed->set_shm_writer(shm_writer);
        feed->run();

        ioc.run();
    } catch (const std::exception& e) {
        std::cerr << "[BINANCE INGESTION ERROR] Excepcion fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
