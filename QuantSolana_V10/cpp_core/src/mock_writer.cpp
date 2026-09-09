#include "shm_writer.hpp"
#include <iostream>
#include <chrono>

using namespace quantsolana;

int main() {
    try {
        SharedMemoryWriter writer("QuantSolana_V10_SHM");
        std::cout << "Mock Writer started. Pumping data into Shared Memory...\n";
        
        OrderbookState state{};
        state.bids[0].qty = 10.5; // Fixed validation value
        
        uint64_t counter = 0;
        double current_price = 150.55;
        bool price_up = true;
        
        while (true) {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            state.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            
            // Oscillate price
            if (price_up) {
                current_price += 0.01;
                if (current_price >= 151.54) price_up = false;
            } else {
                current_price -= 0.01;
                if (current_price <= 150.55) price_up = true;
            }
            
            state.mid_price = current_price;
            state.last_update_id = counter++;
            
            writer.write_state(state);
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    
    return 0;
}
