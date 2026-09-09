#pragma once

#include <array>
#include <cstdint>

namespace quantsolana {

struct Level {
    double price;
    double qty;
};

// Align to 64 bytes (cache line size) to prevent false sharing and cache misses
struct alignas(64) OrderbookState {
    std::array<Level, 10> bids;
    std::array<Level, 10> asks;
    
    double mid_price;
    double imbalance;
    double ofi_t;
    double ofi_ema5;
    double cvd;
    double spread;
    
    uint64_t last_update_id;
    uint64_t timestamp_ms;
};

} // namespace quantsolana
