# QuantumSolana — Autonomous Trading Bot v9

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=for-the-badge&logo=pytorch)
![Binance](https://img.shields.io/badge/Binance-Futures-yellow?style=for-the-badge&logo=binance)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Autonomous quantitative trading system for Solana (SOL/USDT perpetual futures on Binance).**
> Combines Reinforcement Learning, GPU-accelerated optimization, and real-time NLP sentiment analysis to operate 24/7 without human intervention.

---

## Architecture Overview

```
quant_orchestrator.py          ← Master process manager
├── cerebro_cuda.py            ← PSO parameter optimizer (NVIDIA GPU / CUDA)
├── cerebro_sentimiento.py     ← NLP macro sentiment engine (FinBERT)
└── bot_sol/
    ├── bot_sol_paper.py       ← Paper trading simulator (safe mode)
    └── bot_sol_live.py        ← Live trading (real money)

bot_core.py                    ← Shared core: DB, RL inference, Risk Manager
cerebro_rl_env.py              ← Custom Gymnasium RL training environment
cerebro_rl_train.py            ← PPO training script (Stable-Baselines3)
entrenar_filtro_xgb_sol.py     ← XGBoost experience replay filter trainer
```

---

## Key Features

### Reinforcement Learning Brain (PPO)
- **Architecture:** 3-layer MLP [256, 256, 256] trained with Proximal Policy Optimization
- **Action space:** Hold / Open Long / Open Short / Close Position
- **Observation space:** 16 market microstructure features + agent state
- **Training data:** 350,000+ real market ticks from Binance Futures WebSocket
- **Symmetric Data Augmentation:** Trains simultaneously on real + mirror-inverted market to eliminate directional bias
- **Dense Net PnL rewards:** Commission costs are deducted from rewards tick-by-tick, forcing the agent to only enter trades with sufficient expected edge

### Market Microstructure Features
| Feature | Description |
|---------|-------------|
| `imbalance` | Level-2 order book bid/ask volume imbalance |
| `ofi` | Order Flow Imbalance (Cont et al. method) |
| `ofi_ema_5/15` | Short/medium-term OFI exponential moving averages |
| `cvd` | Cumulative Volume Delta from aggTrade stream |
| `liq_longs/shorts` | Real-time liquidation cascade tracking |
| `ema_15m_dist` | Distance from 15-min EMA (trend context) |
| `rsi_5m` | 5-min RSI (momentum context) |
| `vwap_dist` | Distance from intraday VWAP |
| `macro_sentiment` | FinBERT NLP score from live crypto news |
| `spread` | Real-time bid-ask spread |
| `wall_gap` | Depth-10 order book wall distance |

### GPU Optimizer (Cerebro CUDA)
- **Differential Evolution** optimizer (via `scipy.optimize`) with a GPU-accelerated backtesting kernel running on Numba CUDA
- Each candidate set of parameters is evaluated in parallel inside the RTX GPU via a custom `@cuda.jit` kernel — 200 candidates per generation simultaneously
- Continuously optimizes TP%, SL%, OFI thresholds, and IA confidence levels across the full historical tick dataset
- Writes optimal parameters to `config_params.json` — loaded hot by the bot every 30s

### NLP Sentiment Engine (Cerebro Sentimiento)
- FinBERT model loaded on GPU for financial text classification
- Scrapes RSS feeds from CoinDesk, CryptoNews, CryptoPanic in real time
- Outputs a [-1, +1] macro sentiment score injected into the RL observation vector

### Risk Manager
- Dynamic position sizing based on current equity curve
- Reduces leverage automatically during drawdown periods
- Hard stop-loss, take-profit, trailing stop, and time-decay exit logic

---

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU (CUDA 11.8+) — Required for GPU optimizer and sentiment model
- Binance Futures account with API keys

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jedua/QuantumSolana.git
cd QuantumSolana

# 2. Create and activate conda environment
conda create -n cerebro_gpu python=3.10
conda activate cerebro_gpu

# 3. Install dependencies
pip install stable-baselines3[extra] gymnasium torch torchvision
pip install xgboost scikit-learn pandas numpy numba
pip install websockets requests colorama transformers
pip install binance-futures-connector

# 4. Configure your API keys
cp .env.example .env
# Edit .env with your Binance API Key and Secret
```

### Environment Variables (`.env`)
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

---

## Usage

### Step 1 — Collect market data
Start the paper trading bot first to populate the historical database:
```bash
python quant_orchestrator.py
```
Let it run for at least **12-24 hours** to collect enough ticks.

### Step 2 — Train the RL model
```bash
python cerebro_rl_train.py
```
Training takes ~35-60 minutes on an RTX 4070 SUPER for 2,000,000 timesteps.

### Step 3 — (Optional) Train the XGBoost filter
```bash
python entrenar_filtro_xgb_sol.py
```

### Step 4 — Run the full system
```bash
# Paper trading (safe simulation)
python quant_orchestrator.py

# Live trading (real money — use with caution)
# Set MODO_PRODUCCION = True in quant_orchestrator.py first
python quant_orchestrator.py
```

---

## Project Structure

```
QuantumSolana/
├── .env                        # API keys (NOT in git)
├── .gitignore
├── README.md
├── quant_orchestrator.py       # Main entry point
├── bot_core.py                 # Shared core module
├── cerebro_rl_env.py           # RL Gymnasium environment
├── cerebro_rl_train.py         # PPO training script
├── cerebro_cuda.py             # GPU-based PSO optimizer
├── cerebro_sentimiento.py      # NLP sentiment engine
├── entrenar_filtro_xgb_sol.py  # XGBoost filter trainer
└── bot_sol/
    ├── bot_sol_paper.py        # Paper trading bot
    └── bot_sol_live.py         # Live trading bot
```

---

## Disclaimer

> **This software is for educational and research purposes only.**
> Cryptocurrency trading involves substantial risk of loss. Past performance in simulation does not guarantee future results with real money.
> The authors are not responsible for any financial losses incurred through the use of this software.
> **Never trade with money you cannot afford to lose.**

---

## License

MIT License — see [LICENSE](LICENSE) for details.
