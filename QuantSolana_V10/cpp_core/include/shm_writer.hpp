#pragma once

#include "orderbook.hpp"
#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <atomic>
#include <cstring>
#include <iostream>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace quantsolana {

struct SharedMemoryBlock {
    std::atomic<uint64_t> sequence{0};
    char padding[56]; // Explicit padding to force 64-byte alignment for the next struct
    OrderbookState state;
};

class SharedMemoryWriter {
    boost::interprocess::windows_shared_memory shm_;
    boost::interprocess::mapped_region region_;
    SharedMemoryBlock* block_;
    HANDLE event_handle_;

public:
    explicit SharedMemoryWriter(const char* shm_name = "QuantSolana_V10_SHM") {
        using namespace boost::interprocess;
        try {
            shm_ = windows_shared_memory(
                open_or_create, 
                shm_name, 
                read_write, 
                sizeof(SharedMemoryBlock)
            );
        } catch (const interprocess_exception& ex) {
            std::cerr << "Failed to initialize shared memory: " << ex.what() << "\n";
            throw;
        }

        // Map the entire shared memory in this process
        region_ = mapped_region(shm_, read_write);
        block_ = static_cast<SharedMemoryBlock*>(region_.get_address());
        
        // Initialize sequence safely
        block_->sequence.store(0, std::memory_order_relaxed);
        
        // Initialize Windows Named Event for Zero-Latency IPC
        event_handle_ = CreateEventA(NULL, FALSE, FALSE, "QuantSolana_V10_Event");
        if (!event_handle_) {
            std::cerr << "Warning: Failed to create/open QuantSolana_V10_Event. IPC will fallback to polling.\n";
        }
    }

    ~SharedMemoryWriter() {
        if (event_handle_) {
            CloseHandle(event_handle_);
        }
    }

    // Seqlock write logic
    void write_state(const OrderbookState& new_state) {
        // 1. Increment sequence to odd (signaling write in progress)
        uint64_t seq = block_->sequence.load(std::memory_order_relaxed);
        block_->sequence.store(seq + 1, std::memory_order_release);

        // 2. Transfer data via memcpy (bypassing constructors for maximum speed)
        std::memcpy(&block_->state, &new_state, sizeof(OrderbookState));

        // 3. Increment sequence to even (signaling write complete)
        // Memory order release ensures memcpy commits to RAM before sequence updates
        block_->sequence.store(seq + 2, std::memory_order_release);

        // 4. Fire the event to wake up Python instantly
        if (event_handle_) {
            SetEvent(event_handle_);
        }
    }
};

} // namespace quantsolana
