import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import time
import pandas as pd
import numpy as np
import requests
import sqlite3
from colorama import Fore, Style, init
from datetime import datetime
import warnings
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

PARENT_DIR = ROOT_DIR

load_dotenv(os.path.join(PARENT_DIR, ".env"))
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

try:
    binance_client = Client(API_KEY, API_SECRET)
    print("[BINANCE] Cliente inicializado correctamente.")
except Exception as e:
    print(f"[BINANCE ERROR] Error iniciando cliente: {e}")
    sys.exit(1)

def ejecutar_orden_mercado(symbol, side, qty, reduce_only=False):
    try:
        qty_rounded = round(qty, 3)
        order = binance_client.futures_create_order(
            symbol=symbol.upper(),
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=qty_rounded,
            reduceOnly=reduce_only
        )
        print(f"[BINANCE EXITO] Orden {side} ejecutada: {qty_rounded} {symbol} (Reduce: {reduce_only})")
        return True, ""
    except Exception as e:
        print(f"[BINANCE ERROR] Fallo orden {side}: {e}")
        return False, str(e)

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
LIVE_LOG_FILE = os.path.join(PARENT_DIR, "live_trading_log.txt")
LIVE_STATE_FILE = os.path.join(PARENT_DIR, "live_state_sol.json")
PAPER_MODEL_FILE = os.path.join(SCRIPT_DIR, "..", "modelo_rl_sol_v10.zip")
LIVE_TERMINAL_LOG_FILE = os.path.join(PARENT_DIR, "live_log_terminal_data.json")

init(autoreset=True)
warnings.filterwarnings('ignore')

def main_loop(db, ia, exp, risk_manager, filtro_xgb):
    print(f"{Fore.MAGENTA}=====================================================")
    print(f"{Fore.MAGENTA} [SISTEMA] SOLANA BOT V10 - LIVE TRADING (DINERO REAL IPC SHM)")
    print(f"{Fore.MAGENTA}=====================================================")
    
    cargar_configuracion(CONFIG_FILE)
    shm_reader = SharedMemoryReader("QuantSolana_V10_SHM")
    
    regime_detector = MarketRegime()
    posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, confianza_ia = cargar_estado_simulacion(LIVE_STATE_FILE)
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

    def cerrar_posicion(motivo):
        nonlocal posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, cooldown_actual
        if posicion is None: return
        
        st = shm_reader.read_latest_state()
        precio_salida = st.bids[0].price if posicion == 'LONG' else st.asks[0].price
        
        side_order = SIDE_SELL if posicion == 'LONG' else SIDE_BUY
        qty_sol = monto_invertido / precio_salida
        exito, msg = ejecutar_orden_mercado("SOLUSDT", side_order, qty_sol, reduce_only=True)
        
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
        print(f"\n{color}[LIVE TRADE CERRADO] {posicion} | Motivo: {motivo} | PnL: ${pnl_usd:.2f} ({pnl_neto_pct*100:.2f}%) | Acumulado: ${pnl_acumulado:.2f}")
        registrar_trade_log(LIVE_LOG_FILE, posicion, precio_entrada, precio_salida, pnl_usd, pnl_neto_pct, motivo, LIVE_TERMINAL_LOG_FILE)
        
        posicion = None
        precio_entrada = 0.0
        max_pnl_pct = 0.0
        timestamp_entrada = 0
        guardar_estado_simulacion(LIVE_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

    print(f"[SHM IPC LIVE] Conectado exitosamente. Esperando datos en vivo...")

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

            # Deduplicacion: si el uid no cambio, esperar la proxima actualizacion del WebSocket
            current_uid = obs_dict.get('last_update_id', 0)
            if current_uid == last_update_id:
                if time.time() - last_uid_change_time > 30:
                    if not shm_stale_warned:
                        print(f"[WARN] SHM congelada hace 30s (uid={current_uid}). Ingestor C++ posiblemente desconectado.")
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
                macro_sentiment_score = read_sentiment(SENTIMENT_FILE_PATH, LIVE_TERMINAL_LOG_FILE)
                ema_15m_val, rsi_5m_val, atr_5m_val = bot_core.fetch_mtf_data("SOLUSDT", LIVE_TERMINAL_LOG_FILE)
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
                print(f"\n[ESTADO LIVE V10] PnL Neto: ${pnl_acumulado:.2f} | Trades: {trades_totales} | Ticks: {ticks_procesados}")
                ultimo_reporte = now

            # Control de entradas y salidas
            if posicion is None:
                # --- FILTRO DE VOLATILIDAD DINAMICO (Adaptativo al precio) ---
                atr_minimo = max(0.06, mid_price * 0.0012)
                mercado_tiene_volatilidad = cached_atr_5m >= atr_minimo

                if action == 1 and mercado_tiene_volatilidad:
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        action = 0
                    else:
                        confirmaciones_long += 1
                        confirmaciones_short = 0
                        if confirmaciones_long >= 1:
                            precio_entrada = best_ask
                            monto_invertido = MONTO_USDT
                            qty_sol = monto_invertido / precio_entrada
                            exito, msg = ejecutar_orden_mercado("SOLUSDT", SIDE_BUY, qty_sol)
                            if exito:
                                posicion = 'LONG'
                                timestamp_entrada = now
                                confirmaciones_long = 0
                                print(f"\n{Fore.GREEN}[LIVE ENTRY LONG] Entrada al Ask: ${precio_entrada:.2f} | IMB: {obs_dict['imbalance']:.4f} | OFI: {obs_dict['ofi']:.2f}")
                                guardar_estado_simulacion(LIVE_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

                elif action == 2 and mercado_tiene_volatilidad:
                    if not filtro_xgb.aprobar_trade(obs_dict):
                        action = 0
                    else:
                        confirmaciones_short += 1
                        confirmaciones_long = 0
                        if confirmaciones_short >= 1:
                            precio_entrada = best_bid
                            monto_invertido = MONTO_USDT
                            qty_sol = monto_invertido / precio_entrada
                            exito, msg = ejecutar_orden_mercado("SOLUSDT", SIDE_SELL, qty_sol)
                            if exito:
                                posicion = 'SHORT'
                                timestamp_entrada = now
                                confirmaciones_short = 0
                                print(f"\n{Fore.RED}[LIVE ENTRY SHORT] Entrada al Bid: ${precio_entrada:.2f} | IMB: {obs_dict['imbalance']:.4f} | OFI: {obs_dict['ofi']:.2f}")
                                guardar_estado_simulacion(LIVE_STATE_FILE, posicion, precio_entrada, max_pnl_pct, pnl_acumulado, trades_totales, monto_invertido, rachas_perdidas, timestamp_entrada, 0.0)

            else:
                if current_pnl_pct > max_pnl_pct:
                    max_pnl_pct = current_pnl_pct

                if action == 3 and (current_pnl_pct >= 0.0045 or (now - timestamp_entrada > 900)):
                    cerrar_posicion("RL_CLOSE")
                elif max_pnl_pct >= bot_core.TAKE_PROFIT_PCT and (max_pnl_pct - current_pnl_pct) >= (bot_core.TAKE_PROFIT_PCT * 0.25):
                    cerrar_posicion("TAKE_PROFIT/TS")
                elif current_pnl_pct <= -bot_core.STOP_LOSS_PCT:
                    cerrar_posicion("HARD_SL")
                elif now - timestamp_entrada > 2700 and current_pnl_pct <= 0.0005:
                    cerrar_posicion("TIME_DECAY")

            time.sleep(0.001)

        except Exception as e:
            print(f"{Fore.RED}[ERROR HOT LOOP LIVE V10] {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    db = GestorDB(DB_NAME, LIVE_TERMINAL_LOG_FILE)
    ia = CerebroRL(PAPER_MODEL_FILE, LIVE_TERMINAL_LOG_FILE)
    exp = GestorExperiencia(os.path.join(PARENT_DIR, "cerebro_experiencia.db"))
    risk_manager = RiskManager(balance_inicial=100.0, max_leverage=LEVERAGE)
    filtro_xgb = FiltroExperienciaXGB(os.path.join(PARENT_DIR, "filtro_xgb_sol.joblib"))
    
    try:
        main_loop(db, ia, exp, risk_manager, filtro_xgb)
    except KeyboardInterrupt:
        print("\n[SISTEMA] Bot Live V10 detenido de forma segura.")
        db.close()
        exp.close()