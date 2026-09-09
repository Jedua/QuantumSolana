import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import sqlite3
import pandas as pd
from stable_baselines3 import PPO
from cerebro_rl_env import TradingEnv
import numpy as np

def main():
    print("Loading data from DB...")
    conn = sqlite3.connect('cerebro_sol.db')
    df = pd.read_sql_query("SELECT * FROM mercado_ticks ORDER BY id ASC LIMIT 10000", conn)
    conn.close()
    
    print("Loading environment...")
    env = TradingEnv(df)
    print("Loading model...")
    model = PPO.load('modelo_rl_sol_v10.zip', env=env)
    
    obs, _ = env.reset()
    action_counts = {0:0, 1:0, 2:0, 3:0}
    
    print("Running 2000 steps on historical data...")
    for i in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
            
    print(f"Action counts on history: {action_counts}")

if __name__ == '__main__':
    main()
