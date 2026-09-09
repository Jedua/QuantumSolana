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
PAPER_LOG_FILE = os.path.join(PARENT_DIR, "paper_trading_log.txt")
PAPER_STATE_FILE = os.path.join(PARENT_DIR, "sim_state_sol.json")
PAPER_MODEL_FILE = os.path.join(SCRIPT_DIR, "..", "modelo_rl_sol_v10.zip")
PAPER_TERMINAL_LOG_FILE = os.path.join(PARENT_DIR, "log_terminal_data.json")

init(autoreset=True)
warnings.filterwarnings('ignore')

def main_loop(db, ia, exp, risk_manager, filtro_xgb):
    print(f"{Fore.MAGENTA}=====================================================")
    print(f"{Fore.MAGENTA} [SISTEMA] SOLANA BOT V10 - PAPER TRADING (IPC SHM)")
    print(f"{Fore.MAGENTA}=====================================================")
    
    cargar_configuracion(CONFIG_FILE)
    shm_reader = SharedMemoryReader("QuantSolana_V10_SHM")
    
    regime_detector = MarketRegime()
    posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, confianza_ia = cargar_estado_simulacion(PAPER_STATE_FILE)
    cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS if rachas_perdidas > 0 else 0
    
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

    def cerrar_posicion(motivo):
        nonlocal posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, cooldown_actual, obs_dict_entrada
        if posicion is None: return
        
        st = shm_reader.read_latest_state()
        precio_salida = st.bids[0].price if posicion == 'LONG' else st.asks[0].price
        
        pnl_bruto_pct = (precio_salida - precio_entrada) / precio_entrada if posicion == 'LONG' else (precio_entrada - precio_salida) / precio_entrada
        pnl_neto_pct = (pnl_bruto_pct - ROUND_TRIP_FEE) * LEVERAGE
        pnl_usd = monto_invertido * pnl_neto_pct
        
        pnl_acumulado += pnl_usd
        trades_totales += 1
        
        if pnl_usd < 0:
            rachas_perdidas += 1
            cooldown_actual = rachas_perdidas * COOLDOWN_SEGUNDOS
        else:
            rachas_perdidas = 0
            cooldown_actual = 0
            
        color = Fore.GREEN if pnl_usd > 0 else Fore.RED
        print(f"\n{color}[TRADE CERRADO] {posicion} | Motivo: {motivo} | PnL: ${pnl_usd:.2f} ({pnl_neto_pct*100:.2f}%) | Acumulado: ${pnl_acumulado:.2f}")
        registrar_trade_log(PAPER_LOG_FILE, posicion, precio_entrada, precio_salida, pnl_usd, pnl_neto_pct, motivo, PAPER_TERMINAL_LOG_FILE)
        
        # --- Guardar experiencia de cierre para entrenamiento XGBoost ---
        if obs_dict_entrada is not None:
            exp.guardar_snapshot(f"CLOSE_{posicion}", precio_salida, pnl_usd, obs_dict_entrada)
        
        posicion = None
        precio_entrada = 0.0
        max_pnl_pct = 0.0
        timestamp_entrada = 0
        obs_dict_entrada = None
        guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

    print(f"[SHM IPC] Conectado exitosamente. Esperando datos en vivo...")

    last_update_id = 0
    last_uid_change_time = time.time()
    shm_stale_warned = False

    while True:
        try:
            shm_reader.wait_for_update(timeout_ms=100)
            
            try:
                obs_dict = shm_reader.get_obs_dict()
            except Exception:
                continue

            # Deduplicacion: si el uid no cambio, esperar la proxima actualizacion del WebSocket
            current_uid = obs_dict.get('last_update_id', 0)
            if current_uid == last_update_id:
                # Solo advertir si llevamos 30+ segundos sin cambio (SHM realmente muerta)
                if time.time() - last_uid_change_time > 30:
                    if not shm_stale_warned:
                        print(f"[WARN] SHM congelada hace 30s (uid={current_uid}). Ingestor C++ posiblemente desconectado.")
                        shm_stale_warned = True
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

            # Mapeo critico SHM -> modelo RL (nombres distintos)
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

            pos_int = 1 if posicion == 'LONG' else (-1 if posicion == 'SHORT' else 0)
            current_pnl_pct = 0.0
            if pos_int == 1: current_pnl_pct = (best_bid - precio_entrada) / precio_entrada
            elif pos_int == -1: current_pnl_pct = (precio_entrada - best_ask) / precio_entrada

            action = ia.predecir_accion(obs_dict, pos_int, current_pnl_pct)

            if now - ultimo_reporte > 300:
                print(f"\n[ESTADO V10] PnL Neto: ${pnl_acumulado:.2f} | Trades: {trades_totales} | Ticks: {ticks_procesados}")
                ultimo_reporte = now

            # Control de entradas y salidas
            if posicion is None:
                # --- FILTROS CUANTITATIVOS ROBUSTOS ---
                # 1. Filtro de Regimen de Mercado (Evitar shocks de volatilidad destructivos)
                mercado_favorable = regimen_actual != "SHOCK"
                
                # 2. Filtro de Volatilidad Dinámica (Adaptativo al precio de SOL)
                atr_minimo = max(0.06, mid_price * 0.0010)
                mercado_tiene_volatilidad = cached_atr_5m >= atr_minimo
                
                # 3. Filtros de Microestructura (Alineados con config_params.json)
                imb_actual = obs_dict.get('imbalance', 0.0)
                ofi_actual = obs_dict.get('ofi', 0.0)
                ofi_ema5_actual = obs_dict.get('ofi_ema_5', 0.0)
                
                filtro_long_ok = (imb_actual >= bot_core.UMBRAL_IMBALANCE * 0.5) and (ofi_actual >= bot_core.OFI_THRESHOLD)
                filtro_short_ok = (imb_actual <= -bot_core.UMBRAL_IMBALANCE * 0.5) and (ofi_actual <= -bot_core.OFI_THRESHOLD)

                if action == 1 and mercado_favorable and mercado_tiene_volatilidad and filtro_long_ok:
                    # --- FILTRO XGBOOST: Bloquear entrada si el filtro secundario la rechaza ---
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        action = 0
                    else:
                        confirmaciones_long += 1
                        confirmaciones_short = 0
                        if confirmaciones_long >= 1:
                            posicion = 'LONG'
                            precio_entrada = best_ask
                            balance_actual = 25.0 + pnl_acumulado
                            bot_core.LEVERAGE = risk_manager.calcular_apalancamiento(balance_actual)
                            monto_invertido = max(5.0, balance_actual * 0.95)
                            timestamp_entrada = now
                            obs_dict_entrada = obs_dict
                            exp.guardar_snapshot(f"OPEN_{posicion}", precio_entrada, 0.0, obs_dict_entrada)
                            confirmaciones_long = 0
                            print(f"\n{Fore.GREEN}[ENTRY LONG] Entrada al Ask: ${precio_entrada:.2f} | IMB: {obs_dict['imbalance']:.4f} | OFI: {obs_dict['ofi']:.2f} | ATR: {cached_atr_5m:.4f}")
                            guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

                elif action == 2 and mercado_favorable and mercado_tiene_volatilidad and filtro_short_ok:
                    # --- FILTRO XGBOOST: Bloquear entrada si el filtro secundario la rechaza ---
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        action = 0
                    else:
                        confirmaciones_short += 1
                        confirmaciones_long = 0
                        if confirmaciones_short >= 1:
                            posicion = 'SHORT'
                            precio_entrada = best_bid
                            balance_actual = 25.0 + pnl_acumulado
                            bot_core.LEVERAGE = risk_manager.calcular_apalancamiento(balance_actual)
                            monto_invertido = max(5.0, balance_actual * 0.95)
                            timestamp_entrada = now
                            obs_dict_entrada = obs_dict
                            exp.guardar_snapshot(f"OPEN_{posicion}", precio_entrada, 0.0, obs_dict_entrada)
                            confirmaciones_short = 0
                            print(f"\n{Fore.RED}[ENTRY SHORT] Entrada al Bid: ${precio_entrada:.2f} | IMB: {obs_dict['imbalance']:.4f} | OFI: {obs_dict['ofi']:.2f} | ATR: {cached_atr_5m:.4f}")
                            guardar_estado_simulacion(PAPER_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

            else:
                if current_pnl_pct > max_pnl_pct:
                    max_pnl_pct = current_pnl_pct

                # Salidas por estrategia
                if action == 3 and (current_pnl_pct >= 0.0035 or (now - timestamp_entrada > 900)):
                    cerrar_posicion("RL_CLOSE")
                elif max_pnl_pct >= bot_core.TAKE_PROFIT_PCT and (max_pnl_pct - current_pnl_pct) >= (bot_core.TAKE_PROFIT_PCT * 0.35):
                    cerrar_posicion("TAKE_PROFIT/TS")
                elif current_pnl_pct <= -bot_core.STOP_LOSS_PCT:
                    cerrar_posicion("HARD_SL")
                elif now - timestamp_entrada > 3600 and current_pnl_pct <= 0.0005:
                    cerrar_posicion("TIME_DECAY")

        except Exception as e:
            print(f"{Fore.RED}[ERROR HOT LOOP V10] {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    db = GestorDB(DB_NAME, PAPER_TERMINAL_LOG_FILE)
    ia = CerebroRL(PAPER_MODEL_FILE, PAPER_TERMINAL_LOG_FILE)
    exp = GestorExperiencia(os.path.join(PARENT_DIR, "cerebro_experiencia.db"))
    risk_manager = RiskManager(balance_inicial=25.0, max_leverage=LEVERAGE)
    filtro_xgb = FiltroExperienciaXGB(os.path.join(PARENT_DIR, "filtro_xgb_sol.joblib"))
    
    try:
        main_loop(db, ia, exp, risk_manager, filtro_xgb)
    except KeyboardInterrupt:
        print("\n[SISTEMA] Bot Paper V10 detenido de forma segura.")
        db.close()
        exp.close()