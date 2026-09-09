import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

class TradingEnv(gym.Env):
    """
    Entorno de Trading personalizado para OpenAI Gymnasium.
    Simula ejecuciones Taker (pagando spread) con comisiones.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=100.0, leverage=20.0, taker_fee=0.0005):
        super(TradingEnv, self).__init__()
        
        self.df = df
        
        # Fill NaN values for new columns to avoid errors with older historical data
        self.df.fillna({'vwap_dist': 0.0, 'xgb_probability': 0.5}, inplace=True)
        if 'xgb_probability' not in self.df.columns:
            self.df['xgb_probability'] = 0.5
        
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.taker_fee = taker_fee # Comision por lado (0.05%)
        # CORRECTO: taker_fee*2*leverage = 0.02 → breakeven a 0.1% de precio = igual que produccion
        self.round_trip_fee = taker_fee * 2 * leverage
        
        # Acciones:
        # 0: Hold (No hacer nada o mantener posición actual)
        # 1: Open Long (Si ya hay Long, Hold. Si hay Short, cierra Short y abre Long)
        # 2: Open Short (Si ya hay Short, Hold. Si hay Long, cierra Long y abre Short)
        # 3: Close Position (Pasa a Flat)
        self.action_space = spaces.Discrete(4)
        
        # Features del mercado
        self.features_cols = [
            'imbalance', 'spread', 'wall_gap', 'vol_total', 'ofi', 
            'ofi_ema_5', 'ofi_ema_15', 'cvd', 'liq_longs', 'liq_shorts', 
            'ema_15m_dist', 'rsi_5m', 'macro_sentiment', 'vwap_dist',
            'xgb_probability'
        ]
        
        # El observation space incluye features del mercado + estado del agente (posición y PnL)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features_cols) + 2,), dtype=np.float32
        )
        
        # Estado interno
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.trades_count = 0
        self.ticks_in_trade = 0
        self.unrealized_pnl = 0.0
        
    def _get_obs(self):
        # Tomamos la fila actual
        row = self.df.iloc[self.current_step].copy()
        
        # --- NORMALIZACION PARA ESTABILIZAR RED NEURONAL ---
        # Normalizacion Dinamica Asintotica (Tanh)
        # Usamos el volumen total sin logaritmo como factor de escala base
        vol_base = max(row['vol_total'], 1.0)
        
        row['ofi'] = np.tanh(row['ofi'] / vol_base)
        row['ofi_ema_5'] = np.tanh(row['ofi_ema_5'] / vol_base)
        row['ofi_ema_15'] = np.tanh(row['ofi_ema_15'] / vol_base)
        row['cvd'] = np.tanh(row['cvd'] / (vol_base * 10.0))
        
        # Liquidaciones normalizadas (estático suave)
        row['liq_longs'] = np.tanh(row['liq_longs'] / 1000.0)
        row['liq_shorts'] = np.tanh(row['liq_shorts'] / 1000.0)
        
        # Log scaling para valores absolutos gigantes (volumenes)
        row['vol_total'] = np.log1p(row['vol_total'])
        
        # Centrado estandar [-1, 1]
        row['rsi_5m'] = (row['rsi_5m'] - 50.0) / 50.0
        
        obs = row[self.features_cols].values.astype(np.float32)
        
        # Calculamos PnL flotante (Taker)
        current_pnl_pct = 0.0
        if self.position == 1:
            # Salida Long es vendiendo al Bid
            current_pnl_pct = (row['best_bid'] - self.entry_price) / self.entry_price
        elif self.position == -1:
            # Salida Short es comprando al Ask
            current_pnl_pct = (self.entry_price - row['best_ask']) / self.entry_price
            
        # Agregamos variables de estado del bot
        estado_bot = np.array([self.position, current_pnl_pct], dtype=np.float32)
        return np.concatenate((obs, estado_bot))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades_count = 0
        self.ticks_in_trade = 0
        self.unrealized_pnl = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        done = False
        row = self.df.iloc[self.current_step]
        
        best_bid = row['best_bid']
        best_ask = row['best_ask']
        
        # --- LÓGICA DE ACCIONES SIMÉTRICA ---
        
        # 0: Hold (Nada cambia)
        if action == 0:
            pass
            
        # 1: Open Long
        elif action == 1:
            if self.position == -1:
                # Cerrar Short existente primero
                if self.unrealized_pnl > 0: reward += self.unrealized_pnl * 1.0
                else: reward += self.unrealized_pnl * 0.5
                self.position = 0
                self.unrealized_pnl = 0.0
                self.ticks_in_trade = 0
                
            if self.position == 0:
                # Abrir Long (Paga el Ask)
                self.position = 1
                self.entry_price = best_ask
                self.trades_count += 1
                self.unrealized_pnl = 0.0
                
        # 2: Open Short
        elif action == 2:
            if self.position == 1:
                # Cerrar Long existente primero
                if self.unrealized_pnl > 0: reward += self.unrealized_pnl * 1.0
                else: reward += self.unrealized_pnl * 0.5
                self.position = 0
                self.unrealized_pnl = 0.0
                self.ticks_in_trade = 0
                
            if self.position == 0:
                # Abrir Short (Paga el Bid)
                self.position = -1
                self.entry_price = best_bid
                self.trades_count += 1
                self.unrealized_pnl = 0.0
                
        # 3: Close Position
        elif action == 3:
            if self.position == 0:
                reward -= 0.0005  # Castigo minimo por cerrar sin posicion
            else:
                # Materializar PnL al cerrar
                if self.unrealized_pnl > 0:
                    reward += self.unrealized_pnl * 1.2  # Bonus por cerrar en ganancia
                else:
                    reward += self.unrealized_pnl * 0.5  # Penalizacion por perdida
                self.position = 0
                self.unrealized_pnl = 0.0
                self.ticks_in_trade = 0

        # --- DENSE NET PNL (Recompensa continua simétrica por tick) ---
        if self.position != 0:
            current_net = 0.0
            if self.position == 1:
                gross_pnl = (best_bid - self.entry_price) / self.entry_price
                current_net = (gross_pnl * self.leverage) - self.round_trip_fee
            elif self.position == -1:
                gross_pnl = (self.entry_price - best_ask) / self.entry_price
                current_net = (gross_pnl * self.leverage) - self.round_trip_fee
                
            # Recompensa por cambio de PnL (delta reward)
            pnl_delta = current_net - self.unrealized_pnl
            reward += pnl_delta * 2.0  # Amplificar senal de PnL
            self.unrealized_pnl = current_net
            
            # Time decay suave
            self.ticks_in_trade += 1
            reward -= 0.000005
            
            # Stop Loss forzado
            if current_net <= -0.15:
                reward -= 0.1
                self.position = 0
                self.unrealized_pnl = 0.0
                self.ticks_in_trade = 0

            elif self.ticks_in_trade > 3000 and current_net <= 0.001:
                reward -= 0.02
                self.position = 0
                self.unrealized_pnl = 0.0
                self.ticks_in_trade = 0

        # Balance tracking virtual
        self.balance += reward * self.initial_balance

        # --- AVANCE DE TIEMPO ---
        self.current_step += 1
        
        # Comprobar si hemos llegado al final
        if self.current_step >= self.n_steps - 1:
            done = True
            if self.trades_count == 0:
                reward -= 2.0  # Penalizacion severa por no operar
            elif self.trades_count < 5:
                reward -= 0.5  # Penalizacion por operar muy poco
            
        # Comprobar bancarrota
        if self.balance <= self.initial_balance * 0.1:
            done = True
            reward -= 1.0
            
        obs = self._get_obs()
        info = {
            'balance': self.balance,
            'trades': self.trades_count
        }
        
        # Requerimientos de Gymnasium (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info

    def render(self):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Pos: {self.position} | PnL: {self._get_obs()[-1]*100:.2f}%")
