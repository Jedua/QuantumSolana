import mmap
import ctypes
import time
import numpy as np

class Level(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("qty", ctypes.c_double)
    ]

class OrderbookState(ctypes.Structure):
    _fields_ = [
        ("bids", Level * 10),
        ("asks", Level * 10),
        ("mid_price", ctypes.c_double),
        ("imbalance", ctypes.c_double),
        ("ofi_t", ctypes.c_double),
        ("ofi_ema5", ctypes.c_double),
        ("cvd", ctypes.c_double),
        ("spread", ctypes.c_double),
        ("last_update_id", ctypes.c_uint64),
        ("timestamp_ms", ctypes.c_uint64)
    ]

class SharedMemoryBlock(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("padding", ctypes.c_char * 56),
        ("state", OrderbookState)
    ]

class SharedMemoryReader:
    def __init__(self, shm_name="QuantSolana_V10_SHM"):
        self.shm_name = shm_name
        self.shm_size = ctypes.sizeof(SharedMemoryBlock)
        # Windows named shared memory mapping uses fileno -1
        self.mm = mmap.mmap(-1, self.shm_size, tagname=self.shm_name, access=mmap.ACCESS_WRITE)
        
    def read_latest_state(self, max_retries=1000):
        for _ in range(max_retries):
            # 1. Read the initial sequence directly from the start of the buffer
            self.mm.seek(0)
            initial_seq = int.from_bytes(self.mm.read(8), byteorder='little')
            
            # If sequence is odd, C++ is currently writing (Lock-free backoff)
            if initial_seq % 2 != 0:
                continue
                
            # 2. Extract memory block safely using ctypes.from_buffer_copy
            # Note: from_buffer_copy is used to prevent the struct mutating mid-read
            block = SharedMemoryBlock.from_buffer_copy(self.mm)
            
            # 3. Read final sequence from the extracted block
            final_seq = block.sequence
            
            # 4. Verification: If they match, the read is clean and Torn-Read free
            if initial_seq == final_seq:
                return block.state
                
        raise RuntimeError("Demasiados reintentos por Torn Read. Posible bloqueo o desincronizacion severa en C++.")

    def get_obs_dict(self):
        state = self.read_latest_state()
        
        obs = {
            "mid_price": state.mid_price,
            "imbalance": state.imbalance,
            "ofi_t": state.ofi_t,
            "ofi_ema5": state.ofi_ema5,
            "cvd": state.cvd,
            "spread": state.spread,
            "last_update_id": state.last_update_id,
            "timestamp_ms": state.timestamp_ms
        }
        
        obs["bids_price"] = np.array([state.bids[i].price for i in range(10)], dtype=np.float64)
        obs["bids_qty"] = np.array([state.bids[i].qty for i in range(10)], dtype=np.float64)
        obs["asks_price"] = np.array([state.asks[i].price for i in range(10)], dtype=np.float64)
        obs["asks_qty"] = np.array([state.asks[i].qty for i in range(10)], dtype=np.float64)
        
        return obs

    def close(self):
        self.mm.close()
