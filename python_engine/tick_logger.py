import sqlite3
import time
import sys
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PARENT_DIR)
from python_engine.shm_reader import SharedMemoryReader

DB_NAME = os.path.join(PARENT_DIR, "cerebro_sol.db")
BATCH_SIZE = 100
FLUSH_INTERVAL_SEC = 1.0

def main():
    print("[TICK LOGGER V10] Iniciando registrador standalone de alta velocidad...")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # PRAGMAs para máxima velocidad de escritura en paralelo
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    
    # Garantizar la creación de la tabla si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mercado_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            mid_price REAL,
            imbalance REAL,
            spread REAL,
            wall_gap REAL,
            vol_total REAL,
            best_bid REAL,
            best_ask REAL,
            ofi REAL DEFAULT 0.0,
            ofi_ema_5 REAL DEFAULT 0.0,
            ofi_ema_15 REAL DEFAULT 0.0,
            cvd REAL DEFAULT 0.0,
            liq_longs REAL DEFAULT 0.0,
            liq_shorts REAL DEFAULT 0.0,
            ema_15m_dist REAL DEFAULT 0.0,
            rsi_5m REAL DEFAULT 50.0,
            btc_trend REAL DEFAULT 0.0,
            atr_5m REAL DEFAULT 0.0,
            macro_sentiment REAL DEFAULT 0.0,
            vwap_dist REAL DEFAULT 0.0
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_time ON mercado_ticks(timestamp)')
    conn.commit()

    shm_reader = SharedMemoryReader("QuantSolana_V10_SHM")
    
    batch_buffer = []
    last_update_id = 0
    last_flush_time = time.time()

    print("[TICK LOGGER V10] Enganchado a SharedMemoryReader. Registrando datos...")

    while True:
        try:
            try:
                state = shm_reader.read_latest_state()
            except Exception:
                time.sleep(0.001)
                continue

            current_update_id = state.last_update_id
            if current_update_id == 0 and state.bids[0].price > 0:
                current_update_id = int(time.time() * 1000)

            # Filtro de duplicados y validación de datos
            if current_update_id == last_update_id or state.bids[0].price <= 0:
                if batch_buffer and (time.time() - last_flush_time >= FLUSH_INTERVAL_SEC):
                    cursor.executemany('''
                        INSERT INTO mercado_ticks (
                            timestamp, mid_price, imbalance, spread, wall_gap, vol_total, 
                            best_bid, best_ask, ofi, ofi_ema_5, ofi_ema_15, cvd, 
                            liq_longs, liq_shorts, ema_15m_dist, rsi_5m, btc_trend, atr_5m, macro_sentiment
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch_buffer)
                    conn.commit()
                    batch_buffer.clear()
                    last_flush_time = time.time()
                
                time.sleep(0.001)
                continue

            last_update_id = current_update_id
            
            best_bid = state.bids[0].price
            best_ask = state.asks[0].price
            vol_total = state.bids[0].qty + state.asks[0].qty
            wall_gap = state.asks[9].price - state.bids[9].price if state.asks[9].price > 0 else 0.0

            tick_tuple = (
                state.timestamp_ms / 1000.0 if state.timestamp_ms > 0 else time.time(),
                state.mid_price,
                state.imbalance,
                state.spread,
                wall_gap,
                vol_total,
                best_bid,
                best_ask,
                state.ofi_t,
                state.ofi_ema5,
                0.0,
                state.cvd,
                0.0,
                0.0,
                0.0,
                50.0,
                0.0,
                0.0,
                0.0
            )
            
            batch_buffer.append(tick_tuple)

            now = time.time()
            if len(batch_buffer) >= BATCH_SIZE or (now - last_flush_time >= FLUSH_INTERVAL_SEC):
                cursor.executemany('''
                    INSERT INTO mercado_ticks (
                        timestamp, mid_price, imbalance, spread, wall_gap, vol_total, 
                        best_bid, best_ask, ofi, ofi_ema_5, ofi_ema_15, cvd, 
                        liq_longs, liq_shorts, ema_15m_dist, rsi_5m, btc_trend, atr_5m, macro_sentiment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch_buffer)
                conn.commit()
                batch_buffer.clear()
                last_flush_time = now

        except Exception as e:
            print(f"[TICK LOGGER ERROR] {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main()
