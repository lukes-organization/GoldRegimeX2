# Gold Regime X

A hybrid machine learning trading system for **XAUUSD (Gold)** that combines Hidden Markov Models for regime detection with XGBoost for trade signal classification. Designed for live execution through MetaTrader 5 on small cent accounts, with full Telegram remote control and health monitoring.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data Setup](#data-setup)
6. [Configuration](#configuration)
7. [Full Workflow](#full-workflow)
8. [Command Reference](#command-reference)
9. [Signal Logic](#signal-logic)
10. [Risk Management](#risk-management)
11. [Timeframe Configurations](#timeframe-configurations)
12. [Telegram Remote Control](#telegram-remote-control)
13. [MQL5 EA (Alternative Execution)](#mql5-ea-alternative-execution)
14. [Troubleshooting](#troubleshooting)
15. [Security Notes](#security-notes)

---

## How It Works

```
Raw OHLCV CSV
      │
      ▼
Kalman Filter  →  Log Returns (smoothed)
      │
      ▼
GaussianHMM   →  Regime Labels  (Bull=0, Bear=1, Chop=2)
      │
      ▼
XGBoost       →  Signal Probability  [hmm_state, rsi_slope, atr_normalized, prev_log_return]
      │
      ▼
Backtester    →  IS / OOS Sharpe, Drawdown, Trade Count
      │
      ▼
Live Bridge   →  MT5 Market Orders  (IOC fill, ATR-based SL, staged TPs)
```

The optimizer (Optuna) searches across Kalman parameters, HMM states, XGBoost hyperparameters, and the signal probability threshold simultaneously, scoring every trial on **out-of-sample Sharpe only** to prevent overfitting.

---

## Architecture

| File | Purpose |
|------|---------|
| `src/processor.py` | Kalman filter, log returns, RSI, ATR, per-TF config |
| `src/engine_hmm.py` | GaussianHMM training and regime prediction |
| `src/engine_xgb.py` | XGBoost training, ONNX export |
| `src/backtester.py` | Vectorized NumPy backtest with IS/OOS split, session limits, broker costs |
| `src/optimizer.py` | Optuna study with SQLite crash-safe resume, RAM guard, Telegram heartbeat |
| `src/risk_manager.py` | AdaptiveRiskManager, CentConverter, broker cost configs |
| `src/visualizer.py` | 5 chart report: regime overlay, equity curve, features, transition matrix, dashboard |
| `src/mt5_sync.py` | MT5 data downloader |
| `src/validator.py` | Pre-live validation gate (Sharpe threshold check) |
| `src/mt5_trader.py` | Live execution loop: bar detection, feature inference, order placement |
| `src/notifier.py` | Telegram message sender |
| `src/auditor.py` | MT5 deal history report |
| `src/guardian.py` | Multi-TF rolling health monitor |
| `src/remote_control.py` | Telegram long-polling bot for remote commands |
| `main.py` | CLI entry point for all modes |
| `mql5/GoldRegimeX.mq5` | MT5 Expert Advisor with ONNX inference (alternative to Python bridge) |

---

## Prerequisites

- **Windows** (MetaTrader5 Python package is Windows-only)
- **Python 3.11**
- **MetaTrader5 terminal** open and logged into your broker account
- **XAUUSD** visible in Market Watch
- **Algorithmic trading enabled** in MT5: Tools → Options → Expert Advisors → Allow Algorithmic Trading
- A **Telegram bot** (for notifications and remote control — optional but recommended)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/lucasmos/GoldRegime_X.git
cd GoldRegime_X

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
copy .env.example .env
```

---

## Data Setup

The system requires historical OHLCV CSV files exported from MetaTrader5. In MT5:

1. Open the **History Center**: Tools → History Center (or press F2)
2. Select **XAUUSD** → your desired timeframe
3. Click **Export** → save as CSV
4. Copy the files to the `data/raw/` directory with these exact names:

| Timeframe | Expected filename |
|-----------|-------------------|
| H1 | `data/raw/XAU_1h_data.csv` |
| M15 | `data/raw/XAU_15m_data.csv` |
| M5 | `data/raw/XAU_5m_data.csv` |

**Format**: semicolon-delimited, columns: `Date;Open;High;Low;Close;Volume`

> The system works with H1, M15 and M5 timeframes. More data = better optimization. Aim for at least 2 years of history; the included H1 dataset covers 2004–2025.

---

## Configuration

### 1. Create a Telegram Bot (optional but strongly recommended)

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Follow the prompts and copy the **Bot Token**
3. Send `/start` to your new bot
4. Find your **Chat ID** by visiting:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   Look for `"chat": {"id": 123456789}` in the response.
5. Get your **User ID** via [@userinfobot](https://t.me/userinfobot)

### 2. Edit your `.env` file

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ALLOWED_USER_ID=your_telegram_user_id_here

# Defaults used when START TRADING is pressed from Telegram
LIVE_TF=H1
LIVE_BROKER=headway_cent
LIVE_BALANCE=15
```

> If you skip Telegram setup, the system works fine — notifications are simply no-ops.

---

## Full Workflow

### Step 1 — Process raw data

```bash
python main.py --mode process --tf H1
```

Applies Kalman filter, computes log returns, RSI, ATR, and saves a processed parquet file. Repeat for each timeframe you want to trade.

### Step 2 — Optimize hyperparameters

```bash
python main.py --mode optimize --trials 300 --broker headway_cent --balance 15 --tf H1
```

Runs an Optuna study that searches across:
- Kalman filter parameters (`obs_cov`, `trans_cov`)
- HMM state count (`n_states`: 2–4)
- Signal probability threshold (`prob_threshold`: 0.52–0.68)
- XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`)

Each trial is scored on **OOS Sharpe only** to prevent IS data leakage. Trials with fewer than 50 OOS trades are discarded. Progress is saved to `models/study.db` after every trial — safe to interrupt with Ctrl+C and resume later.

**To resume after interruption**, run the exact same command. Optuna detects the existing study and picks up where it left off.

**To start fresh**, delete `models/study.db` manually before running.

### Step 3 — Train final model

```bash
python main.py --mode train --broker headway_cent --balance 15 --tf H1
```

Trains HMM and XGBoost using the best parameters found by the optimizer. Prints IS/OOS Sharpe, drawdown, and trade count. Saves `models/hmm_model.pkl` and `models/xgb_model.pkl`.

### Step 4 — Compare timeframes (optional)

```bash
python main.py --mode compare --broker headway_cent --balance 15 --tf H1,M15
```

Side-by-side OOS performance for multiple timeframes. Supports any comma-separated combination (e.g. `M5,M15,H1`).

### Step 5 — Export ONNX model (for MQL5 EA)

```bash
python main.py --mode export
```

Converts the XGBoost model to `models/xgb_model.onnx` for use with the MQL5 Expert Advisor. Required only if you want to use the EA instead of the Python bridge.

### Step 6 — Generate visual report

```bash
python main.py --mode report --broker headway_cent --balance 15 --tf H1
```

Saves 5 charts to `reports/H1/`:
1. Regime overlay on price
2. Equity curve (IS vs OOS)
3. Feature analysis
4. HMM transition matrix
5. Summary dashboard

### Step 7 — Validate before going live

```bash
python main.py --mode sync_validate --period 3m --broker headway_cent --balance 15 --tf H1
```

Downloads the last 3 months of live MT5 data, runs the full pipeline, and checks recent Sharpe ratio.

| Status | Sharpe | Action |
|--------|--------|--------|
| **PASS** | ≥ 0.8 | Proceed to live |
| **WARN** | 0.5–0.8 | Proceed with caution or re-optimise |
| **FAIL** | < 0.5 | Re-optimize and retrain before going live |

FAIL exits with code 1 and blocks the live script.

### Step 8 — Paper trade first

```bash
python main.py --mode live --account demo --broker headway_cent --balance 15 --tf H1
```

Logs every signal with full detail but sends no orders to MT5. Run for a few days to verify signals look correct before going live.

### Step 9 — Go live

```bash
python main.py --mode live --account live --broker headway_cent --balance 15 --tf H1
```

You will be prompted to type `YES` to confirm. The loop then:
1. Detects each newly completed bar (polls every 5 seconds)
2. Fetches 200 bars of OHLCV data for warm-up
3. Runs Kalman → HMM → XGBoost inference
4. Applies session limits, margin check, and spread viability guard
5. Places IOC market orders with ATR-based SL and staged TPs

**Important**: Remove the GoldRegimeX.mq5 EA from the XAUUSD chart before starting the Python bridge. Both use `MAGIC_NUMBER = 123456` — running both simultaneously will double-count daily trades.

---

## Command Reference

```
python main.py --mode <MODE> [OPTIONS]
```

| Mode | Description |
|------|-------------|
| `process` | Process raw CSV → parquet |
| `optimize` | Run / resume Optuna hyperparameter search |
| `train` | Train HMM + XGBoost with best params |
| `compare` | Side-by-side OOS comparison across TFs |
| `export` | Export XGBoost → ONNX |
| `report` | Generate 5-chart visual report |
| `sync_validate` | Download live data + validate model health |
| `live` | Run the live execution loop |
| `audit` | Generate and send daily MT5 deal report |
| `guardian` | Continuous rolling Sharpe health monitor |
| `listen` | Start Telegram remote control bot |

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tf` | `H1` | Timeframe: `H1`, `M15`, `M5` (or comma-separated for compare) |
| `--broker` | `standard` | Broker profile: `headway_cent`, `standard` |
| `--balance` | `15` | Account size in USD (used for lot sizing and risk tier) |
| `--trials` | `250` | Number of Optuna trials (optimize mode) |
| `--period` | `3m` | Lookback period for sync/validate: `3m`, `6m`, `12m` |
| `--account` | `live` | `demo` (dry run, no orders) or `live` (real orders) |
| `--interval` | `3600` | Guardian check interval in seconds |

---

## Signal Logic

A trade fires when ALL of the following are true:

| Condition | BUY | SELL |
|-----------|-----|------|
| XGBoost probability | `prob > prob_threshold` | `prob < (1 - prob_threshold)` |
| HMM regime | Not Chop (state ≠ 2) | Not Chop (state ≠ 2) |
| Session limit | Under daily cap | Under daily cap |
| No open position | No existing GRX position | No existing GRX position |
| Margin check | Sufficient free margin | Sufficient free margin |
| Spread viability (M5) | TP1 distance ≥ 1.5× spread | TP1 distance ≥ 1.5× spread |

**`prob_threshold` is tuned by the optimizer per TF/study** and loaded automatically at live startup. If no optimized value exists, the hardcoded TF defaults are used (H1: 0.65, M15: 0.65, M5: 0.70).

### Staged Take-Profits

Each signal places multiple positions with different TPs (SL distance = ATR × 2.0):

| Regime | TP1 | TP2 (Runner) |
|--------|-----|--------------|
| Bull / Bear — M5 | 1.0 × SL | 3.0 × SL |
| Bull / Bear — M15/H1 | 1.5 × SL | 3.0 × SL |
| Chop — all TFs | 0.8 × SL | None (single position) |

When TP1 fills, the runner's stop-loss is automatically moved to break-even. If the regime shifts to Chop while a runner is active, the runner is closed immediately at market.

### Deviation (Slippage Tolerance)

| Condition | M5 | M15/H1 |
|-----------|----|--------|
| Normal | 30 pts | 20 pts |
| High-vol (HMM self-transition < 0.70) | 50 pts | 50 pts |

---

## Risk Management

Position sizing uses a fixed 1% risk rule per position:

```
lot_size = (1% × account_balance_USD) / (ATR(14) × 2.0)
```

Minimum lot is 0.01 (micro-lot). All lots are rounded to 2 decimal places.

**Daily trade limits (adaptive):**

| Account Balance | Regime | Max Trades/Day | Positions/Signal |
|----------------|--------|----------------|-----------------|
| ≤ $50 | Any | 2 | 1 |
| > $50 | Bull / Bear | 3 | 2 |
| > $50 | Chop | 2 | 2 |

**Cent Account (Headway):** MT5 displays $15 USD as `1500.00`. Pass `--broker headway_cent` and the bridge divides the raw balance by 100 automatically.

---

## Timeframe Configurations

| Parameter | M5 | M15 | H1 |
|-----------|-----|-----|-----|
| Kalman `obs_cov` default | 0.05 | 4.0 | 1.0 |
| Bars/day (annualization) | 288 | 96 | 24 |
| TP1 multiplier (trending) | 1.0× SL | 1.5× SL | 1.5× SL |
| Base deviation | 30 pts | 20 pts | 20 pts |
| Spread viability guard | Yes | No | No |
| 5-day readiness gate | Yes | No | No |

**M5 readiness gate:** `models/m5_meta.json` must exist and be less than 120 hours old. The optimizer creates this file automatically after completing an M5 study. If stale, `--mode live --tf M5` will exit with an error until you re-optimize.

**M5 workflow:**
```bash
python main.py --mode process --tf M5
python -c "import optuna; optuna.delete_study('gold_regime_x_small_headway_cent_M5', 'sqlite:///models/study.db')"
python main.py --mode optimize --tf M5 --trials 300 --broker headway_cent --balance 15
python main.py --mode train --tf M5 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M5 --period 3m --broker headway_cent --balance 15
python main.py --mode live --tf M5 --account demo --broker headway_cent --balance 15
```

---

## Telegram Remote Control

Start the listener alongside your trading session:

```bash
python main.py --mode listen --broker headway_cent --balance 15
```

A keyboard appears in your Telegram chat:

| Button | Action |
|--------|--------|
| 🚀 START TRADING | Launches the live loop using `.env` defaults |
| 🛑 STOP TRADING | Kills the running live loop process |
| 📉 START OPTIMIZE (M5) | Starts/resumes M5 Optuna study |
| 📊 BOT STATUS | Last 24h P&L, win rate, and floating positions |

A nightly summary is automatically sent to your chat at **23:55 UTC** while the listener runs.

### Guardian — Continuous health monitor

```bash
python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600 --broker headway_cent --balance 15
```

Checks rolling Sharpe for each TF every hour. Sends a Telegram alert if any TF drops below 0.6.

### Audit — On-demand deal report

```bash
python main.py --mode audit --broker headway_cent --balance 15
```

Prints and sends the last 24h deal summary with P&L breakdown.

### Parallel Optimization

Open multiple terminals and run the same optimize command in each — they share `study.db` safely via SQLite locking:

```bash
# Terminal 1
python main.py --mode optimize --tf M5 --trials 500 --broker headway_cent --balance 15
# Terminal 2 (same command)
python main.py --mode optimize --tf M5 --trials 500 --broker headway_cent --balance 15
```

---

## MQL5 EA (Alternative Execution)

`mql5/GoldRegimeX.mq5` is a fully self-contained MetaTrader5 Expert Advisor that:
- Loads `models/xgb_model.onnx` directly inside MT5
- Replicates the same regime → signal → risk logic in MQL5
- Supports cent accounts via the `IsCentAccount` input
- Uses the same `MAGIC_NUMBER = 123456`

**Do not run the EA and the Python bridge simultaneously** — they share the magic number and will double-count trades.

To use the EA:
1. Run `--mode export` to generate `models/xgb_model.onnx`
2. Copy `mql5/GoldRegimeX.mq5` and `models/xgb_model.onnx` to your MT5 `MQL5/Experts/` folder
3. Compile in MetaEditor (F7)
4. Attach to the XAUUSD chart

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: models/hmm_model.pkl` | Models not trained | Run `--mode train` |
| `FileNotFoundError: models/m5_meta.json` | M5 not optimized | Run `--mode optimize --tf M5` |
| Validation FAIL every day | Model too old for current regime | Re-optimize and retrain |
| `Order failed: retcode=10006` | No broker connection | Check MT5 connection indicator |
| `Order failed: retcode=10015` | Price moved past deviation | Will retry next bar; try a slower TF |
| `Insufficient margin` repeated | Account below minimum for lot | Top up account or pass a lower `--balance` |
| No trades firing all session | prob_threshold too strict for model | Re-optimize — the optimizer will tune the threshold |
| Signals logged but no MT5 orders | Running in demo mode | Switch to `--account live` |
| Double positions appearing | EA still attached to chart | Remove GoldRegimeX.mq5 EA from XAUUSD chart |
| Telegram errors in log | No internet or wrong token | Check `.env` credentials; revoke and regenerate from @BotFather if exposed |

**Emergency stop:** Press **Ctrl+C** in the terminal. The loop handles `KeyboardInterrupt` cleanly, logs the shutdown, and disconnects from MT5. All open positions remain open — close manually from the MT5 Trade tab.

---

## Security Notes

- **Never commit `.env`** — it contains your live Telegram token and is listed in `.gitignore`
- **`.env.example` contains only placeholders** — fill in your real values in `.env` only
- `ALLOWED_USER_ID` is the single security gate for Telegram remote commands — only your Telegram ID can issue them
- If credentials are accidentally committed, immediately revoke the bot token via @BotFather `/revoke` and generate a new one
- The listener uses Telegram's long-polling API — no public webhook or open port needed

---

## State Labels

These are hardcoded across all Python modules and the MQL5 EA — do not change:

| Label | Integer | Meaning |
|-------|---------|---------|
| Bull | 0 | Trending upward |
| Bear | 1 | Trending downward |
| Chop | 2 | Sideways / low conviction |
