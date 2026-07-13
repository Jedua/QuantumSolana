import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from colorama import Fore, Style, init
import joblib

init(autoreset=True)

DB_NAME = "cerebro_experiencia.db"
MODEL_NAME = "filtro_xgb_sol.joblib"

def entrenar_filtro():
    print(f"{Fore.CYAN}[INFO] Cargando datos de experiencia desde {DB_NAME}...")
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM snapshots_rl", conn)
        conn.close()
    except Exception as e:
        print(f"{Fore.RED}[ERROR] No se pudo leer la base de datos: {e}")
        return

    if df.empty:
        print(f"{Fore.RED}[ERROR] No hay suficientes datos en {DB_NAME}.")
        return

    df_open = df[df['accion_tomada'].str.startswith('OPEN')].reset_index(drop=True)
    df_close = df[df['accion_tomada'].str.startswith('CLOSE')].reset_index(drop=True)
    
    # Asegurarnos de que tengan la misma longitud
    min_len = min(len(df_open), len(df_close))
    df_open = df_open.iloc[:min_len]
    df_close = df_close.iloc[:min_len]

    features_cols = [
        'imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 
        'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 
        'ema_15m_dist', 'rsi_5m', 'macro_sentiment', 'vwap_dist'
    ]

    X = df_open[features_cols]
    
    # Target: 1 si el trade fue ganador (pnl > 0), 0 si fue perdedor (pnl <= 0)
    y = (df_close['pnl_resultado'] > 0).astype(int)

    # Imprimir balance de clases
    print(f"{Fore.YELLOW}[INFO] Trades totales analizados: {len(y)}")
    print(f"Ganadores (1): {y.sum()}")
    print(f"Perdedores (0): {len(y) - y.sum()}")

    if len(y) < 10:
        print(f"{Fore.RED}[ERROR] Se necesitan al menos 10 trades para entrenar el filtro.")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"{Fore.CYAN}[INFO] Entrenando modelo XGBoost...")
    
    # Usar class_weight para balancear si hay mas perdedores que ganadores
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{Fore.GREEN}[SUCCESS] Filtro entrenado. Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_NAME)
    print(f"{Fore.GREEN}[SUCCESS] Modelo guardado como {MODEL_NAME}.")

if __name__ == "__main__":
    entrenar_filtro()
