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
MODEL_PATH = "modelo_rl_sol" # stable-baselines3 agrega el .zip automáticamente
TERMINAL_LOG_PATH = "log_terminal_data.json"

def recalcular_features(df):
    """Reconstruye las features de OFI y EMAs en caso de que falten en la BD antigua"""
    for col in ['cvd', 'liq_longs', 'liq_shorts', 'ema_15m_dist', 'rsi_5m', 'macro_sentiment', 'vwap_dist']:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)
        
    # Recalcular VWAP si está en ceros (datos antiguos)
    if (df['vwap_dist'] == 0.0).all():
        df['pv'] = df['mid_price'] * df['vol_total']
        cumulative_pv = df['pv'].cumsum()
        cumulative_v = df['vol_total'].cumsum()
        vwap = cumulative_pv / cumulative_v
        df['vwap_dist'] = (df['mid_price'] - vwap) / vwap
        df = df.drop(columns=['pv'])

    if 'ofi' not in df.columns or 'ofi_ema_5' not in df.columns:
        df['vol_bid'] = df['vol_total'] * (1 + df['imbalance']) / 2
        df['vol_ask'] = df['vol_total'] * (1 - df['imbalance']) / 2
        df['e_b'] = 0.0
        df['e_a'] = 0.0
        
        df.loc[df['best_bid'] > df['best_bid'].shift(1), 'e_b'] = df['vol_bid']
        df.loc[df['best_bid'] == df['best_bid'].shift(1), 'e_b'] = df['vol_bid'] - df['vol_bid'].shift(1)
        df.loc[df['best_bid'] < df['best_bid'].shift(1), 'e_b'] = -df['vol_bid'].shift(1)
        
        df.loc[df['best_ask'] < df['best_ask'].shift(1), 'e_a'] = df['vol_ask']
        df.loc[df['best_ask'] == df['best_ask'].shift(1), 'e_a'] = df['vol_ask'] - df['vol_ask'].shift(1)
        df.loc[df['best_ask'] > df['best_ask'].shift(1), 'e_a'] = -df['vol_ask'].shift(1)
        
        df['ofi'] = df['e_b'] - df['e_a']
        
        span_5, span_15 = 5, 15
        alpha_5, alpha_15 = 2 / (span_5 + 1), 2 / (span_15 + 1)
        df['ofi_ema_5'] = 0.0
        df['ofi_ema_15'] = 0.0
        
        if len(df) > 0:
            df.loc[0, 'ofi_ema_5'] = df.loc[0, 'ofi']
            df.loc[0, 'ofi_ema_15'] = df.loc[0, 'ofi']
            for i in range(1, len(df)):
                df.loc[i, 'ofi_ema_5'] = alpha_5 * df.loc[i, 'ofi'] + (1 - alpha_5) * df.loc[i-1, 'ofi_ema_5']
                df.loc[i, 'ofi_ema_15'] = alpha_15 * df.loc[i, 'ofi'] + (1 - alpha_15) * df.loc[i-1, 'ofi_ema_15']

    return df.dropna().reset_index(drop=True)

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
    print(f"\n{Fore.YELLOW}[TRAIN] Construyendo modelo PPO...")
    # Usamos MlpPolicy (Red Neuronal Perceptrón Multicapa)
    # Aumentamos la arquitectura a 3 capas de 256 neuronas
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,       # Más contexto por batch (era 2048)
        batch_size=256,
        n_epochs=10,
        gamma=0.97,          # Priorizar recompensas cercanas (scalping)
        ent_coef=0.06,       # Exploración agresiva para romper Hold-Collapse
        clip_range=0.1,      # Aprendizaje más conservador y estable
        policy_kwargs=dict(net_arch=[256, 256, 256]), # CEREBRO MÁS GRANDE
        device="auto",
        tensorboard_log="./tensorboard_rl_logs/"
    )
    
    if os.path.exists(MODEL_PATH + ".zip"):
        # Al cambiar la arquitectura de la red (net_arch), el modelo viejo es matematicamente incompatible.
        # Renombramos el archivo viejo como backup y empezamos fresco para curar el sesgo Long.
        backup_name = MODEL_PATH + "_backup_long_bias.zip"
        if os.path.exists(backup_name):
            os.remove(backup_name)
        os.rename(MODEL_PATH + ".zip", backup_name)
        print(f"{Fore.BLUE}[INFO] Modelo sesgado respaldado como {backup_name}. Empezando multiverso desde cero.")
        
    print(f"\n{Fore.GREEN}[TRAIN] Iniciando entrenamiento (2,000,000 timesteps)...")
    try:
        model.learn(total_timesteps=2000000, progress_bar=True)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[WARN] Entrenamiento interrumpido por el usuario.")
        
    print(f"\n{Fore.GREEN}[OK] Entrenamiento finalizado. Guardando modelo en {MODEL_PATH}")
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
