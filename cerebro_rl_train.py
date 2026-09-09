import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import pandas as pd
import numpy as np
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Por favor instala las dependencias de RL: pip install stable-baselines3[extra] gymnasium")
    sys.exit(1)
from colorama import Fore, Style, init

from cerebro_rl_env import TradingEnv
from bot_core import GestorDB

init(autoreset=True)

DB_NAME = "cerebro_sol.db"
MODEL_PATH = "modelo_rl_sol_v10" # stable-baselines3 agrega el .zip automáticamente
TERMINAL_LOG_PATH = "log_terminal_data.json"

def recalcular_features(df):
    """Reconstruye y calcula vectorialmente las features técnicas y microestructurales."""
    # Asegurar existencia de columnas base
    cols_base = ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'macro_sentiment', 'vwap_dist', 'atr_5m', 'btc_trend']
    for col in cols_base:
        if col not in df.columns: 
            df[col] = 0.0
        else: 
            df[col] = df[col].fillna(0.0)

    # 1. RSI 5m (Rolling window de 300 ticks ~ 5 minutos de flujo continuo)
    if (df['rsi_5m'] == 50.0).all() or (df['rsi_5m'].std() == 0):
        delta = df['mid_price'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=300, min_periods=30, adjust=False).mean()
        avg_loss = loss.ewm(span=300, min_periods=30, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-6)
        df['rsi_5m'] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    # 2. EMA 15m dist (span 900 ticks)
    if (df['ema_15m_dist'] == 0.0).all() or (df['ema_15m_dist'].std() == 0):
        ema_15m = df['mid_price'].ewm(span=900, adjust=False).mean()
        df['ema_15m_dist'] = ((df['mid_price'] - ema_15m) / ema_15m).fillna(0.0)

    # 3. ATR 5m aproximado sobre ticks (span 300)
    if 'atr_5m' not in df.columns or (df['atr_5m'] == 0.0).all():
        high = df['best_ask']
        low = df['best_bid']
        hl = high - low
        df['atr_5m'] = hl.rolling(window=300, min_periods=10).mean().bfill().fillna(0.1)

    # 4. VWAP dist (Rolling window 1800 ticks ~ 30 minutos de session)
    if (df['vwap_dist'] == 0.0).all() or (df['vwap_dist'].std() == 0):
        pv = df['mid_price'] * df['vol_total']
        cum_pv = pv.rolling(window=1800, min_periods=50).sum()
        cum_vol = df['vol_total'].rolling(window=1800, min_periods=50).sum()
        vwap = cum_pv / cum_vol.replace(0, 1e-6)
        df['vwap_dist'] = ((df['mid_price'] - vwap) / vwap).fillna(0.0)

    # 5. OFI y EMAs vectorizadas al 100% (sin bucles for de Python)
    if 'ofi' not in df.columns or (df['ofi'] == 0.0).all():
        vol_bid = df['vol_total'] * (1 + df['imbalance']) / 2.0
        vol_ask = df['vol_total'] * (1 - df['imbalance']) / 2.0
        
        b_bid = df['best_bid']
        b_bid_prev = b_bid.shift(1).bfill()
        v_bid_prev = vol_bid.shift(1).bfill()
        
        e_b = pd.Series(0.0, index=df.index)
        e_b = np.where(b_bid > b_bid_prev, vol_bid,
              np.where(b_bid == b_bid_prev, vol_bid - v_bid_prev, -v_bid_prev))
        
        b_ask = df['best_ask']
        b_ask_prev = b_ask.shift(1).bfill()
        v_ask_prev = vol_ask.shift(1).bfill()
        
        e_a = pd.Series(0.0, index=df.index)
        e_a = np.where(b_ask < b_ask_prev, vol_ask,
              np.where(b_ask == b_ask_prev, vol_ask - v_ask_prev, -v_ask_prev))
              
        df['ofi'] = e_b - e_a

    if 'ofi_ema_5' not in df.columns or (df['ofi_ema_5'] == 0.0).all():
        df['ofi_ema_5'] = df['ofi'].ewm(span=5, adjust=False).mean()
        
    if 'ofi_ema_15' not in df.columns or (df['ofi_ema_15'] == 0.0).all():
        df['ofi_ema_15'] = df['ofi'].ewm(span=15, adjust=False).mean()

    # Sanitización de infinitos y NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mid_price', 'best_bid', 'best_ask']).reset_index(drop=True)
    return df

def main():
    print(f"{Fore.MAGENTA}=====================================================")
    print(f"{Fore.MAGENTA}  MOTOR RL INICIADO (STABLE-BASELINES3 + PYTORCH)    ")
    print(f"{Fore.MAGENTA}=====================================================")
    
    print(f"{Fore.CYAN}[INFO] Cargando datos históricos de {DB_NAME}...")
    db = GestorDB(DB_NAME, TERMINAL_LOG_PATH)
    # Cargar suficientes ticks para el entrenamiento (Aumentado a 2M)
    df = db.obtener_datos_entrenamiento(2000000)
    db.close()
    
    if df.empty or len(df) < 1000:
        print(f"{Fore.RED}[ERROR] No hay suficientes datos en la BD para entrenar.")
        sys.exit(1)
        
    # Limpiar posibles NaNs de las transformaciones y recalcular OFI si falta
    df = recalcular_features(df)
    
    # --- DATA AUGMENTATION (CURA DEL SESGO DIRECCIONAL) ---
    print(f"{Fore.CYAN}[INFO] Creando multiverso de mercado (Normal + Invertido) para eliminar sesgos...")
    df_inv = df.copy()
    
    # Invertir features direccionales
    df_inv['imbalance'] *= -1
    df_inv['ofi'] *= -1
    df_inv['ofi_ema_5'] *= -1
    df_inv['ofi_ema_15'] *= -1
    df_inv['cvd'] *= -1
    if 'ema_15m_dist' in df_inv.columns: df_inv['ema_15m_dist'] *= -1
    if 'vwap_dist' in df_inv.columns: df_inv['vwap_dist'] *= -1
    if 'macro_sentiment' in df_inv.columns: df_inv['macro_sentiment'] *= -1
    if 'btc_trend' in df_inv.columns: df_inv['btc_trend'] *= -1
    
    # Rsi invertido (100 - rsi)
    if 'rsi_5m' in df_inv.columns: df_inv['rsi_5m'] = 100.0 - df_inv['rsi_5m']
    
    # Liquidaciones cruzadas (las liquidaciones de toros ahora son de osos)
    if 'liq_longs' in df_inv.columns and 'liq_shorts' in df_inv.columns:
        df_inv['liq_longs'], df_inv['liq_shorts'] = df['liq_shorts'].copy(), df['liq_longs'].copy()
    
    # Invertir Precios
    p0 = df['mid_price'].iloc[0]
    df_inv['mid_price'] = p0 - (df['mid_price'] - p0)
    spread = df['best_ask'] - df['best_bid']
    df_inv['best_bid'] = df_inv['mid_price'] - (spread / 2)
    df_inv['best_ask'] = df_inv['mid_price'] + (spread / 2)
    
    print(f"{Fore.GREEN}[SUCCESS] Datos procesados: {len(df)} ticks x2 Entornos.")
    
    # Crear los entornos
    env_normal = TradingEnv(df)
    env_invertido = TradingEnv(df_inv)
    
    # Vectorizar ambos entornos: la IA entrena en un universo alcista y uno bajista SIMULTANEAMENTE
    vec_env = DummyVecEnv([lambda: env_normal, lambda: env_invertido])
    
    # Configurar el Agente PPO
    print(f"\n{Fore.YELLOW}[TRAIN] Construyendo modelo PPO optimizado para Scalping HFT...")
    # Usamos MlpPolicy con capas balanceadas para inferencia ultra-rapida
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.015,       # Exploracion equilibrada sin forzar sobreoperacion
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="auto",
        tensorboard_log="./tensorboard_rl_logs/"
    )
    
    if os.path.exists(MODEL_PATH + ".zip"):
        backup_name = MODEL_PATH + "_backup_hold_collapse.zip"
        if os.path.exists(backup_name):
            os.remove(backup_name)
        os.rename(MODEL_PATH + ".zip", backup_name)
        print(f"{Fore.BLUE}[INFO] Modelo anterior respaldado como {backup_name}. Entrenando desde cero con reward corregido.")
        
    print(f"\n{Fore.GREEN}[TRAIN] Iniciando entrenamiento (2,000,000 timesteps)...")
    try:
        model.learn(total_timesteps=2000000, progress_bar=True)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[WARN] Entrenamiento interrumpido por el usuario.")
        
    print(f"\n{Fore.GREEN}[OK] Entrenamiento finalizado. Guardando modelo en {MODEL_PATH}")
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
