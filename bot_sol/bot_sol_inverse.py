import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

PARENT_DIR = ROOT_DIR

import time
import pandas as pd
import numpy as np
import requests
import sqlite3
from colorama import Fore, Style, init
from datetime import datetime
import warnings

from python_engine.shm_reader import SharedMemoryReader
from bot_core import (
    cargar_configuracion, log_terminal_event, registrar_trade_log,
    guardar_estado_simulacion, cargar_estado_simulacion, MarketRegime,
    GestorDB, CerebroRL, read_sentiment, GestorExperiencia, RiskManager,
    FiltroExperienciaXGB, MONTO_USDT, LEVERAGE, MAX_PERDIDA_DIARIA,
    COOLDOWN_SEGUNDOS, HORA_INICIO_OP, HORA_FIN_OP, ROUND_TRIP_FEE
)
import bot_core

DB_NAME = os.path.join(PARENT_DIR, "cerebro_sol.db")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "..", "config_params.json")
PAPER_LOG_FILE = os.path.join(PARENT_DIR, "paper_trading_inverse_log.txt")
PAPER_STATE_FILE = os.path.join(PARENT_DIR, "sim_state_sol_inverse.json")
PAPER_MODEL_FILE = os.path.join(SCRIPT_DIR, "..", "modelo_rl_sol_v10.zip")
PAPER_TERMINAL_LOG_FILE = os.path.join(PARENT_DIR, "log_terminal_inverse_data.json")

init(autoreset=True)
warnings.filterwarnings('ignore')

def main_loop(db, ia, exp, risk_manager, filtro_xgb):
    print(f"{Fore.CYAN}==================================================================")
    print(f"{Fore.CYAN} [BOT TRAMPA ESPEJO INVERSO V10] ESPECULO CON EL FRACASO DEL MODELO")
    print(f"{Fore.CYAN}  * Cada Virtual HARD_SL/TIME_DECAY se convierte en GANANCIA REAL *")
    print(f"{Fore.CYAN}==================================================================")
    
    cargar_configuracion(CONFIG_FILE)
    shm_reader = SharedMemoryReader("QuantSolana_V10_SHM")
    
    regime_detector = MarketRegime()
    posicion_real, precio_entrada_real, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, confianza_ia = cargar_estado_simulacion(PAPER_STATE_FILE)
    cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS if rachas_perdidas > 0 else 0
    
    posicion_virtual = 'LONG' if posicion_real == 'SHORT' else ('SHORT' if posicion_real == 'LONG' else None)
    precio_entrada_virtual = precio_entrada_real
    max_pnl_virtual_pct = 0.0
    
    confirmaciones_long = 0
    confirmaciones_short = 0
    
    ticks_procesados = 0
    ultimo_reporte = time.time()
    ultima_recarga_config = time.time()
    
    macro_sentiment_score = 0.0
    last_sentiment_read = 0
    cached_ema_15m = 0.0
    cached_rsi_5m = 50.0
    cached_atr_5m = 0.0
    SENTIMENT_FILE_PATH = os.path.join(SCRIPT_DIR, "..", "macro_sentiment.json")
    obs_dict_entrada = None

    def cerrar_posicion_espejo(motivo_virtual):
        nonlocal posicion_real, posicion_virtual, precio_entrada_real, precio_entrada_virtual, max_pnl_pct, max_pnl_virtual_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, cooldown_actual, obs_dict_entrada
        if posicion_real is None: return
        
        st = shm_reader.read_latest_state()
        precio_salida_real = st.asks[0].price if posicion_real == 'SHORT' else st.bids[0].price
        
        # PnL REAL de la posición invertida
        if posicion_real == 'SHORT':
            pnl_bruto_real = (precio_entrada_real - precio_salida_real) / precio_entrada_real
        else:
            pnl_bruto_real = (precio_salida_real - precio_entrada_real) / precio_entrada_real
            
        pnl_neto_real = (pnl_bruto_real - ROUND_TRIP_FEE) * LEVERAGE
        pnl_usd_real = monto_invertido * pnl_neto_real
        
        pnl_acumulado += pnl_usd_real
        trades_totales += 1
        
        if pnl_usd_real < 0:
            rachas_perdidas += 1
            cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS
        else:
            rachas_perdidas = 0
            cooldown_actual = 0
            
        color = Fore.GREEN if pnl_usd_real > 0 else Fore.RED
        resultado_str = "GANANCIA (VIRTUAL_SL_INVERTIDO)" if pnl_usd_real > 0 else "PERDIDA (VIRTUAL_TP_INVERTIDO)"
        print(f"\n{color}[ESPEJO CERRADO] Real {posicion_real} (Virtual {posicion_virtual}) | Motivo Virtual: {motivo_virtual} | {resultado_str} | PnL Real: ${pnl_usd_real:.2f} ({pnl_neto_real*100:.2f}%) | Acumulado: ${pnl_acumulado:.2f}")
        
        registrar_trade_log(PAPER_LOG_FILE, f"REAL_{posicion_real}_(MODEL_{posicion_virtual})", precio_entrada_real, precio_salida_real, pnl_usd_real, pnl_neto_real, f"ESPEJO_{motivo_virtual}", PAPER_TERMINAL_LOG_FILE)
        
        if obs_dict_entrada is not None:
            exp.guardar_snapshot(f"CLOSE_INVERSE_{posicion_real}", precio_salida_real, pnl_usd_real, obs_dict_entrada)
        
        posicion_real = None
        posicion_virtual = None
        precio_entrada_real = 0.0
        precio_entrada_virtual = 0.0
        max_pnl_pct = 0.0
        max_pnl_virtual_pct = 0.0
        timestamp_entrada = 0
        obs_dict_entrada = None
        guardar_estado_simulacion(PAPER_STATE_FILE, posicion_real, precio_entrada_real, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

    print(f"[SHM IPC ESPEJO] Conectado exitosamente. Esperando datos en vivo...")

    last_update_id = 0
    last_uid_change_time = time.time()
    shm_stale_warned = False

    while True:
        try:
            try:
                obs_dict = shm_reader.get_obs_dict()
            except Exception:
                time.sleep(0.001)
                continue

            current_uid = obs_dict.get('last_update_id', 0)
            if current_uid == last_update_id:
                if time.time() - last_uid_change_time > 30:
                    if not shm_stale_warned:
                        print(f"[WARN] SHM congelada hace 30s (uid={current_uid}). Ingestor C++ desconectado.")
                        shm_stale_warned = True
                time.sleep(0.001)
                continue
            last_update_id = current_uid
            last_uid_change_time = time.time()
            if shm_stale_warned:
                print(f"[OK] SHM reconectada. Datos frescos recibidos (uid={current_uid}).")
                shm_stale_warned = False
                
            now = time.time()
            if now - ultima_recarga_config > 30:
                cargar_configuracion(CONFIG_FILE)
                ultima_recarga_config = now

            if now - last_sentiment_read > 60:
                macro_sentiment_score = read_sentiment(SENTIMENT_FILE_PATH, PAPER_TERMINAL_LOG_FILE)
                ema_15m_val, rsi_5m_val, atr_5m_val = bot_core.fetch_mtf_data("SOLUSDT", PAPER_TERMINAL_LOG_FILE)
                if ema_15m_val is not None:
                    cached_ema_15m = ema_15m_val
                    cached_rsi_5m = rsi_5m_val
                    cached_atr_5m = atr_5m_val
                last_sentiment_read = now

            obs_dict['macro_sentiment'] = macro_sentiment_score
            obs_dict['vol_total'] = obs_dict['bids_qty'][0] + obs_dict['asks_qty'][0]
            obs_dict['best_bid'] = obs_dict['bids_price'][0]
            obs_dict['best_ask'] = obs_dict['asks_price'][0]
            obs_dict['wall_gap'] = obs_dict['asks_price'][9] - obs_dict['bids_price'][9]
            obs_dict['rsi_5m'] = cached_rsi_5m
            obs_dict['xgb_probability'] = 0.5

            obs_dict['ofi'] = obs_dict.get('ofi_t', 0.0)
            obs_dict['ofi_ema_5'] = obs_dict.get('ofi_ema5', 0.0)
            obs_dict['ofi_ema_15'] = 0.0
            obs_dict['liq_longs'] = 0.0
            obs_dict['liq_shorts'] = 0.0
            mid_price = obs_dict['mid_price']
            obs_dict['ema_15m_dist'] = (mid_price - cached_ema_15m) / cached_ema_15m if cached_ema_15m > 0 else 0.0
            obs_dict['btc_trend'] = 0.0
            obs_dict['atr_5m'] = cached_atr_5m
            obs_dict['vwap_dist'] = 0.0

            ticks_procesados += 1
            best_bid = obs_dict['best_bid']
            best_ask = obs_dict['best_ask']
            mid_price = obs_dict['mid_price']

            regimen_actual = regime_detector.update(mid_price)

            pos_int_virtual = 1 if posicion_virtual == 'LONG' else (-1 if posicion_virtual == 'SHORT' else 0)
            current_pnl_virtual_pct = 0.0
            if pos_int_virtual == 1: current_pnl_virtual_pct = (best_bid - precio_entrada_virtual) / precio_entrada_virtual
            elif pos_int_virtual == -1: current_pnl_virtual_pct = (precio_entrada_virtual - best_ask) / precio_entrada_virtual

            model_action = ia.predecir_accion(obs_dict, pos_int_virtual, current_pnl_virtual_pct)

            if now - ultimo_reporte > 300:
                print(f"\n[ESTADO ESPEJO V10] PnL Neto Real: ${pnl_acumulado:.2f} | Trades: {trades_totales} | Ticks: {ticks_procesados}")
                ultimo_reporte = now

            # Control de entradas y salidas
            if posicion_real is None:
                atr_minimo = max(0.06, mid_price * 0.0012)
                mercado_tiene_volatilidad = cached_atr_5m >= atr_minimo
                
                # --- LÓGICA ESPEJO TRAMPA ---
                # Modelo pide LONG (action == 1) -> Nosotros abrimos SHORT real al Bid
                # Modelo pide SHORT (action == 2) -> Nosotros abrimos LONG real al Ask
                if model_action == 1 and mercado_tiene_volatilidad:
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        pass
                    else:
                        confirmaciones_short += 1
                        confirmaciones_long = 0
                        if confirmaciones_short >= 1:
                            posicion_virtual = 'LONG'
                            posicion_real = 'SHORT'
                            precio_entrada_virtual = best_ask
                            precio_entrada_real = best_bid
                            monto_invertido = MONTO_USDT
                            timestamp_entrada = now
                            obs_dict_entrada = obs_dict
                            exp.guardar_snapshot(f"OPEN_REAL_{posicion_real}", precio_entrada_real, 0.0, obs_dict_entrada)
                            confirmaciones_short = 0
                            print(f"\n{Fore.YELLOW}[ESPEJO TRAMPA: Modelo pidió LONG -> Abrimos SHORT REAL] Entry Bid: ${precio_entrada_real:.2f} (Virtual Ask: ${precio_entrada_virtual:.2f})")
                            guardar_estado_simulacion(PAPER_STATE_FILE, posicion_real, precio_entrada_real, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

                elif model_action == 2 and mercado_tiene_volatilidad:
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        pass
                    else:
                        confirmaciones_long += 1
                        confirmaciones_short = 0
                        if confirmaciones_long >= 1:
                            posicion_virtual = 'SHORT'
                            posicion_real = 'LONG'
                            precio_entrada_virtual = best_bid
                            precio_entrada_real = best_ask
                            monto_invertido = MONTO_USDT
                            timestamp_entrada = now
                            obs_dict_entrada = obs_dict
                            exp.guardar_snapshot(f"OPEN_REAL_{posicion_real}", precio_entrada_real, 0.0, obs_dict_entrada)
                            confirmaciones_long = 0
                            print(f"\n{Fore.CYAN}[ESPEJO TRAMPA: Modelo pidió SHORT -> Abrimos LONG REAL] Entry Ask: ${precio_entrada_real:.2f} (Virtual Bid: ${precio_entrada_virtual:.2f})")
                            guardar_estado_simulacion(PAPER_STATE_FILE, posicion_real, precio_entrada_real, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

            else:
                if current_pnl_virtual_pct > max_pnl_virtual_pct:
                    max_pnl_virtual_pct = current_pnl_virtual_pct

                # Salidas basadas en la POSICIÓN VIRTUAL del Modelo
                # Cada que el modelo VIRTUAL pierde (HARD_SL o TIME_DECAY negativo), nuestra posición REAL GANA.
                if model_action == 3 and (current_pnl_virtual_pct >= 0.0045 or (now - timestamp_entrada > 900)):
                    cerrar_posicion_espejo("RL_CLOSE_VIRTUAL")
                elif max_pnl_virtual_pct >= bot_core.TAKE_PROFIT_PCT and (max_pnl_virtual_pct - current_pnl_virtual_pct) >= (bot_core.TAKE_PROFIT_PCT * 0.50):
                    cerrar_posicion_espejo("TAKE_PROFIT_VIRTUAL_(PERDIDA_REAL)")
                elif current_pnl_virtual_pct <= -bot_core.STOP_LOSS_PCT:
                    # ¡AQUÍ ESTÁ EL TRUCO! El modelo virtual tocó Stop Loss (-0.40%), por ende nuestra posición REAL tocó GANANCIA (+0.40% neto)!
                    cerrar_posicion_espejo("HARD_SL_VIRTUAL_(GANANCIA_REAL)")
                elif now - timestamp_entrada > 5400 and current_pnl_virtual_pct <= 0.0005:
                    cerrar_posicion_espejo("TIME_DECAY_VIRTUAL_(GANANCIA_REAL)")

            time.sleep(0.001)

        except Exception as e:
            print(f"{Fore.RED}[ERROR HOT LOOP ESPEJO V10] {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    db = GestorDB(DB_NAME, PAPER_TERMINAL_LOG_FILE)
    ia = CerebroRL(PAPER_MODEL_FILE, PAPER_TERMINAL_LOG_FILE)
    exp = GestorExperiencia(os.path.join(PARENT_DIR, "cerebro_experiencia.db"))
    risk_manager = RiskManager(balance_inicial=100.0, max_leverage=LEVERAGE)
    filtro_xgb = FiltroExperienciaXGB(os.path.join(PARENT_DIR, "filtro_xgb_sol.joblib"))
    
    try:
        main_loop(db, ia, exp, risk_manager, filtro_xgb)
    except KeyboardInterrupt:
        print("\n[SISTEMA] Bot Espejo Trampa detenido de forma segura.")
        db.close()
        exp.close()
