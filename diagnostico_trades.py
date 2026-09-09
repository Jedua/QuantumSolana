"""
Diagnostico V4: Lee datos REALES del SHM con deduplicacion por uid,
y muestra exactamente que ve el modelo RL y que predice.
"""
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import time
import numpy as np
from python_engine.shm_reader import SharedMemoryReader
from bot_core import cargar_configuracion, CerebroRL
import bot_core

CONFIG_FILE = os.path.join(ROOT_DIR, "config_params.json")
MODEL_FILE = os.path.join(ROOT_DIR, "modelo_rl_sol_v10.zip")

def main():
    cargar_configuracion(CONFIG_FILE)
    shm = SharedMemoryReader("QuantSolana_V10_SHM")
    ia = CerebroRL(MODEL_FILE, None)
    
    action_names = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    last_uid = 0
    unique_ticks = 0
    
    print("=== DIAGNOSTICO V4 (solo ticks unicos, datos reales SHM) ===\n")
    
    start_time = time.time()
    
    while unique_ticks < 200:
        try:
            obs_dict = shm.get_obs_dict()
        except Exception:
            time.sleep(0.001)
            continue
        
        current_uid = obs_dict.get('last_update_id', 0)
        if current_uid == last_uid:
            time.sleep(0.001)
            continue
        last_uid = current_uid
        
        if obs_dict['mid_price'] <= 0:
            continue
        
        # Misma preparacion que bot_sol_paper.py
        obs_dict['macro_sentiment'] = 0.0
        obs_dict['vol_total'] = obs_dict['bids_qty'][0] + obs_dict['asks_qty'][0]
        obs_dict['best_bid'] = obs_dict['bids_price'][0]
        obs_dict['best_ask'] = obs_dict['asks_price'][0]
        obs_dict['wall_gap'] = obs_dict['asks_price'][9] - obs_dict['bids_price'][9]
        obs_dict['rsi_5m'] = 50.0
        obs_dict['xgb_probability'] = 0.5
        
        obs_dict['ofi'] = obs_dict.get('ofi_t', 0.0)
        obs_dict['ofi_ema_5'] = obs_dict.get('ofi_ema5', 0.0)
        obs_dict['ofi_ema_15'] = 0.0
        obs_dict['liq_longs'] = 0.0
        obs_dict['liq_shorts'] = 0.0
        obs_dict['ema_15m_dist'] = 0.0
        obs_dict['btc_trend'] = 0.0
        obs_dict['atr_5m'] = 0.0
        obs_dict['vwap_dist'] = 0.0
        
        # Normalizacion identica a la inferencia
        vol_base = max(obs_dict['vol_total'], 1.0)
        ofi_n = np.tanh(obs_dict['ofi'] / vol_base)
        imb = obs_dict['imbalance']
        
        action = ia.predecir_accion(obs_dict, 0, 0.0)
        action_counts[action] += 1
        unique_ticks += 1
        
        if unique_ticks <= 30 or action != 0:
            print(f"Tick {unique_ticks:4d} | mid={obs_dict['mid_price']:.2f} | imb={imb:+.4f} | ofi={obs_dict['ofi']:+.1f} | ofi_n={ofi_n:+.4f} | vol={obs_dict['vol_total']:.0f} | spread={obs_dict['spread']:.4f} | cvd={obs_dict['cvd']:.1f} | action={action_names[action]}")
    
    elapsed = time.time() - start_time
    print(f"\n=== RESUMEN ({unique_ticks} ticks unicos en {elapsed:.1f}s) ===")
    for a, name in action_names.items():
        pct = (action_counts[a] / unique_ticks * 100) if unique_ticks > 0 else 0
        print(f"  {name:5s}: {action_counts[a]:5d} ({pct:5.1f}%)")
    
    if action_counts[1] == 0 and action_counts[2] == 0:
        print(f"\n  [PROBLEMA] El modelo NUNCA predice LONG/SHORT en {unique_ticks} ticks unicos.")
        print(f"  -> Probando con variaciones sinteticas del imbalance...")
        
        # Tomar el ultimo obs_dict y variar el imbalance artificialmente
        for test_imb in [-0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8]:
            obs_dict['imbalance'] = test_imb
            for test_ofi_n in [-0.5, 0.0, 0.5]:
                obs_dict['ofi'] = test_ofi_n * vol_base  # Desnormalizar
                action = ia.predecir_accion(obs_dict, 0, 0.0)
                if action != 0:
                    print(f"    imb={test_imb:+.1f} ofi_n={test_ofi_n:+.1f} -> {action_names[action]}")
        
        # Prueba exhaustiva de combinaciones
        print(f"\n  -> Busqueda exhaustiva de condiciones de trade...")
        found = 0
        for imb in np.arange(-1.0, 1.01, 0.1):
            for ofi_n in np.arange(-1.0, 1.01, 0.2):
                for spread in [0.01, 0.05, 0.1, 0.5]:
                    obs_dict['imbalance'] = imb
                    obs_dict['ofi'] = ofi_n * vol_base
                    obs_dict['spread'] = spread
                    action = ia.predecir_accion(obs_dict, 0, 0.0)
                    if action != 0:
                        found += 1
                        if found <= 20:
                            print(f"    imb={imb:+.1f} ofi_n={ofi_n:+.1f} spread={spread:.2f} -> {action_names[action]}")
        print(f"  Total combinaciones con trade: {found} de {21*11*4} ({found/(21*11*4)*100:.1f}%)")

if __name__ == "__main__":
    main()
