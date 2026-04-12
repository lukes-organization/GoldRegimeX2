# Gold Regime X

A hybrid machine learning trading system for **XAUUSD (Gold)** that combines Hidden Markov Models for regime detection with XGBoost for trade signal classification. Designed for live execution through MetaTrader 5 on both **Headway Cent** (micro) and **Standard** accounts, with full Telegram remote control and health monitoring.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Account Types — Cent vs Standard](#account-types--cent-vs-standard)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Data Setup](#data-setup)
7. [Configuration](#configuration)
8. [Full Workflow](#full-workflow)
9. [Command Reference](#command-reference)
10. [Signal Logic](#signal-logic)
11. [Risk Management](#risk-management)
12. [Timeframe Configurations](#timeframe-configurations)
13. [Telegram Remote Control](#telegram-remote-control)
14. [MQL5 EA (Alternative Execution)](#mql5-ea-alternative-execution)
15. [Walk-Forward Analysis & Staleness Gate](#walk-forward-analysis--staleness-gate)
16. [Troubleshooting](#troubleshooting)
17. [Security Notes](#security-notes)

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
XGBoost       →  Three-Model Volatility Ensemble
               Features: [hmm_state, rsi_slope, atr_normalized, prev_log_return, usdchf_log_return*]
               Low ATR bucket  → XGBoost model (quiet market)
               Med ATR bucket  → XGBoost model (normal market)
               High ATR bucket → XGBoost model (volatile market)
               * optional — requires data/processed/USDCHF_master.csv (run --mode consolidate first)
      │
      ▼
Backtester    →  IS / OOS Sharpe, Drawdown, Trade Count
      │
      ▼
Live Bridge   →  MT5 Market Orders  (IOC fill, ATR-based SL, staged TPs)
```

The optimizer (Optuna) searches across Kalman parameters, HMM states, XGBoost hyperparameters, and the signal probability thresholds simultaneously, scoring every trial on **out-of-sample Sharpe only** to prevent overfitting.

---

## Architecture

| File | Purpose |
|------|---------|
| `src/processor.py` | Kalman filter, log returns, RSI, ATR, per-TF config |
| `src/engine_hmm.py` | GaussianHMM training and regime prediction |
| `src/engine_xgb.py` | XGBoost training, ONNX export |
| `src/backtester.py` | Vectorized NumPy backtest with IS/OOS split, session limits, broker costs |
| `src/optimizer.py` | Optuna study with per-broker SQLite crash-safe resume, RAM guard, Telegram heartbeat |
| `src/risk_manager.py` | AdaptiveRiskManager, CentConverter, broker cost configs |
| `src/visualizer.py` | 5-chart report: regime overlay, equity curve, features, transition matrix, dashboard |
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

## Account Types — Cent vs Standard

Choose your account type before running any command. The `--broker` flag controls
lot sizing, P&L conversion, spread guards, and probability thresholds throughout
the entire pipeline (optimization, training, live trading).

### Headway Cent Account (`--broker headway_cent`)

A cent account converts your real USD deposit into "cents" displayed 100× larger
in the MT5 terminal. This makes small accounts easier to manage and reduces
per-trade risk to 1/100th of a standard account.

| Item | Cent Account | Example |
|------|-------------|---------|
| Real deposit | $15 USD | Wired to Headway |
| MT5 balance display | 1500.00 USC | (real USD × 100) |
| Minimum lot | 0.01 | = 0.01 oz gold |
| P&L per $1 gold move at 0.01 lot | 0.01 USC | = **$0.0001 real USD** |
| Trade history shows `+15.00` | ÷ 100 | = **$0.15 real USD** |
| Bridge balance handling | Divides raw balance by 100 automatically | Pass `--balance 15` (real USD) |
| Spread viability guard (M5) | TP1 ≥ 1.5× spread | |

**Best for:** Learning the system, verifying that signals and lot sizing work correctly with real broker execution, and starting with minimal capital at risk. All gains and losses are 1/100th of what a standard account would show.

---

### Headway Standard Account (`--broker standard`)

A standard account operates at full contract size in real USD. What you see in MT5 is what you earn or lose.

| Item | Standard Account | Example |
|------|-----------------|---------|
| Minimum recommended balance | $15+ USD | Same as cent for testing |
| Minimum lot | 0.01 | = 1 oz gold |
| P&L per $1 gold move at 0.01 lot | **$1.00 real USD** | 100× more than cent |
| Trade history shows `+2.40` | = **$2.40 real USD** | No conversion needed |
| Bridge balance handling | Uses raw balance directly | Pass `--balance 15` |
| Spread viability guard | TP1 ≥ 3.0× spread | Applied on **all timeframes** |
| Positions per signal (≤ $50) | 2 (each 0.01 lots, floored) | Minimal notional per position |

**Best for:** Scaling to meaningful P&L once the strategy is proven on cent. The stricter 3× spread guard prevents entering trades where the spread eats too much of the expected move.

> **Key practical difference:** On a cent account a trade shown as `+$2.40` in MT5 history is actually `$0.024` real USD. On a standard account it is exactly `$2.40`. The bridge automatically handles cent conversion — the P&L it logs and sends to Telegram is always in **real USD**, regardless of account type.

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

### Getting deep history from MT5 (any symbol, any timeframe)

Since you are already using MT5, you can download historical data directly — provided you adjust your terminal settings to allow for 10 years of bars.

**Action Required:**
- Go to **Tools → Options → Charts**. Change **"Max bars in chart"** to a very high number (e.g., `99,999,999` or just type `unlimited`).
- Go to **View → Symbols** (or `Ctrl+U`). Search for the symbol you want (e.g., **XAUUSD** or **USDCHF**).
- Select the **Bars** tab. Choose your timeframe (M5, M15, H1, etc.) and click **Request**.
- Keep clicking **"Request"** or scroll back the chart until the desired start year is reached.
- Once loaded, click **Export Bars** to save as a `.csv`.

> **Pro Tip:** If your broker (Headway) doesn't have 10 years of history, open a free **MetaQuotes-Demo** account inside MT5. MetaQuotes usually provides much deeper history than individual brokers.

### Keep your data fresh

The processor filters to the **last 10 years anchored at the end of your CSV** (not today's date). If your CSV ends in December 2025, the 20% OOS window closes at December 2025 — but `sync_validate` pulls **live MT5 data up to today**. Any months after your CSV ends will be out-of-distribution for the model.

**Export fresh data from MT5 before each full pipeline re-run** to keep the OOS window aligned with the current market.

### USDCHF — cross-asset USD strength feature (optional, recommended)

USDCHF is used as an intraday DXY proxy (high correlation, natively available on Headway). Each trading timeframe uses a **matching-frequency USDCHF master file** so bars are aligned during feature merging.

#### File naming convention

| Trading TF | Source file in `data/raw/` | Master produced |
|------------|---------------------------|-----------------|
| H1  | `USDCHF_H1.csv` | `data/processed/USDCHF_master.csv` |
| M15 | `USDCHF_M15_<dates>.csv` | `data/processed/USDCHF_master_M15.csv` |
| M5  | `USDCHF_M5_<dates>.csv`  | `data/processed/USDCHF_master_M5.csv` |

The date range in the filename is optional — anything matching `USDCHF_M15_*.csv` or `USDCHF_M5_*.csv` is picked up automatically.

#### How to get USDCHF data free from MT5

Follow the same deep-history export process described in [Getting deep history from MT5](#getting-deep-history-from-mt5-any-symbol-any-timeframe) above — just search for **USDCHF** instead of XAUUSD and select the matching timeframe (M5, M15, or H1). Save each file to `data/raw/` using the naming convention in the table above.

#### Build the master files

```bash
python main.py --mode consolidate
```

This runs all three consolidations in one pass and prints a summary. The pipeline adds USDCHF automatically as a 5th feature per timeframe. If a master is absent for a given TF, that TF degrades gracefully to the 4-feature model.

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

### Step 0 — Consolidate USDCHF (if you have USDCHF CSV exports)

```bash
python main.py --mode consolidate
```

Builds three per-TF master files from the matching source files in `data/raw/`:

| TF | Source | Output |
|----|--------|--------|
| H1  | `USDCHF_H1.csv` | `data/processed/USDCHF_master.csv` |
| M15 | `USDCHF_M15_*.csv` | `data/processed/USDCHF_master_M15.csv` |
| M5  | `USDCHF_M5_*.csv`  | `data/processed/USDCHF_master_M5.csv` |

Run once before processing. TFs without a source file are skipped with a warning and fall back to the 4-feature model.

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
- HMM state count (`n_states`: 3–4 for H1/M15; `{2, 4}` for M5)
- Signal probability thresholds (`prob_threshold`, `short_threshold` — tuned independently)
- XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`)

Each trial is scored on **OOS Sharpe only** to prevent IS data leakage. Trials with degenerate HMM (any state self-transition < 0.80) are discarded. Progress is saved to a **per-broker SQLite database** after every trial — safe to interrupt with Ctrl+C and resume later:

| Broker | Study database |
|--------|---------------|
| `headway_cent` | `models/study_headway_cent.db` |
| `standard` | `models/study_standard.db` |

The two databases are completely independent. You can optimize both accounts without interference.

**To resume after interruption**, run the exact same command. `--trials N` is a **total target** — if 200 trials already exist and you pass `--trials 300`, it runs 100 more.

**To start completely fresh**, delete only the target broker database:
```bash
del models\study_headway_cent.db   # cent only — standard study untouched
del models\study_standard.db       # standard only
```

### Step 3 — Train final model

```bash
python main.py --mode train --broker headway_cent --balance 15 --tf H1
```

Trains HMM and the three-model XGBoost volatility ensemble using the best optimizer parameters. Prints IS/OOS Sharpe, drawdown, and trade count. Saves **broker- and TF-specific model files**:

```
models/hmm_model_H1_headway_cent.pkl
models/xgb_ensemble_H1_headway_cent.pkl
```

Each broker+TF combination gets its own files — training M15 on standard never overwrites H1 on cent.

> Training will **abort** if the resulting HMM is degenerate (any state self-transition < 0.70). This prevents saving a garbage model. If this happens, it means the optimizer study is missing or stale — re-run `--mode optimize` first.

### Step 4 — Compare timeframes (optional)

```bash
python main.py --mode compare --broker headway_cent --balance 15 --tf H1,M15
```

Side-by-side OOS performance for multiple timeframes. Supports any comma-separated combination (e.g. `M5,M15,H1`).

### Step 5 — Export ONNX model (for MQL5 EA)

```bash
python main.py --mode export --tf H1 --broker headway_cent
```

Converts the XGBoost volatility ensemble to ONNX format for use with the MQL5 Expert Advisor. Required only if you want to use the EA instead of the Python bridge.

### Step 6 — Generate visual report

```bash
python main.py --mode report --broker headway_cent --balance 15 --tf H1
```

Saves 5 charts to `reports/H1_headway_cent/`:
1. Regime overlay on price
2. Equity curve (IS vs OOS)
3. Feature analysis
4. HMM transition matrix
5. Summary dashboard

Reports are saved to a **broker- and TF-specific folder** so cent and standard charts never overwrite each other:

```
reports/
├── H1_headway_cent/      ← H1 cent charts
├── H1_standard/          ← H1 standard charts
├── M15_headway_cent/
├── M5_headway_cent/
└── M5_standard/
```

### Step 7 — Validate before going live

```bash
python main.py --mode sync_validate --period 3m --broker headway_cent --balance 15 --tf H1
```

Downloads the last 3 months of live MT5 data, runs the full inference pipeline, and checks recent Sharpe ratio.

| Status | Sharpe | Action |
|--------|--------|--------|
| **PASS** | ≥ 0.8 | Proceed to live |
| **WARN** | 0.5–0.8 | Proceed with caution or re-optimise |
| **FAIL** | < 0.5 | Re-optimize and retrain before going live |

FAIL exits with code 1 and blocks the live script.

> Use `--period 6m` or `--period 8m` for H1 — a 3-month window produces very few H1 trades (42 trades vs 300+ needed), making the Sharpe estimate unreliable.

### Step 8 — Test on demo account

Log MT5 into your **demo** account, then run:

```bash
python main.py --mode demo --tf H1 --broker headway_cent --balance 15
```

`--mode demo` sends **real orders to MT5** on whatever account MT5 is currently logged into. Use a demo account to verify that signals, lot sizes, session limits, and TP/SL logic all behave correctly before risking real money.

> There is no paper trading / simulation mode. All testing is done on a real demo MT5 account so execution behaviour (fills, slippage, spread) is exactly as in live.

### Step 9 — Go live

Log MT5 into your **live** account, then run:

```bash
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```

You will be prompted to type `YES` to confirm. This is the **only difference** from `--mode demo` — both modes send real orders. After confirming, the loop:

1. Detects each newly completed bar (polls every 5 seconds)
2. Fetches 200 bars of OHLCV data for Kalman/HMM warm-up
3. Runs Kalman → HMM → XGBoost inference
4. Applies session limits, margin check, and spread viability guard
5. Places IOC market orders with ATR-based SL and staged TPs
6. Logs closed P&L in real USD with pip movement after every trade

**Important**: Remove the GoldRegimeX.mq5 EA from the XAUUSD chart before starting the Python bridge. Both use `MAGIC_NUMBER = 123456` — running both simultaneously causes the bridge to see the EA's positions as its own, blocking all signal generation and logging no P&L. The bridge logs a `CONFLICT` warning at startup if it detects existing positions with that magic number.

---

## Command Reference

```
python main.py --mode <MODE> [OPTIONS]
```

| Mode | Description |
|------|-------------|
| `consolidate` | Merge all `*USDCHF*.csv` files in `data/raw/` into the USDCHF master |
| `process` | Process raw CSV → parquet |
| `optimize` | Run / resume Optuna hyperparameter search |
| `train` | Train HMM + XGBoost with best params; saves broker+TF-specific model files |
| `compare` | Side-by-side OOS comparison across TFs |
| `export` | Export XGBoost ensemble → ONNX |
| `report` | Generate 5-chart visual report saved to `reports/<TF>_<broker>/` |
| `sync_validate` | Download live MT5 data + validate model health |
| `demo` | Connect to MT5 demo account and run the live execution loop (no YES prompt) |
| `live` | Connect to MT5 live account and run the live execution loop (requires YES) |
| `audit` | Generate and send daily MT5 deal report |
| `guardian` | Continuous rolling Sharpe health monitor |
| `listen` | Start Telegram remote control bot |

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tf` | `H1` | Timeframe: `H1`, `M15`, `M5` (or comma-separated for compare/guardian) |
| `--broker` | `standard` | Broker profile: `headway_cent` or `standard` |
| `--balance` | `15` | Account size in **real USD** — used for lot sizing and risk tier |
| `--trials` | `250` | Total target Optuna trials (optimize mode) |
| `--period` | `3m` | Lookback period for sync/validate: `3m`, `6m`, `12m` |
| `--interval` | `3600` | Guardian check interval in seconds |
| `--prob_threshold` | (from Optuna) | Override BUY probability threshold for live |
| `--short_threshold` | (from Optuna) | Override SELL probability threshold for live |
| `--profit_target` | 4.0 on M5, off elsewhere | Quick-profit close threshold in USD (per position). Pass `0` to disable on M5 |

> **Note:** `--balance` is always in **real USD**. For a Headway Cent account with a $15 deposit, pass `--balance 15` — the bridge handles the ×100 display conversion internally.

---

## Signal Logic

A trade fires when **all** of the following are true:

| Condition | BUY | SELL |
|-----------|-----|------|
| XGBoost probability | `prob > prob_threshold` | `prob < short_threshold` |
| HMM regime (regime-aligned) | **Bull state (0) only** | **Bear state (1) only** |
| Chop state (2 or 3) | Blocked — no signal ever | Blocked — no signal ever |
| Session limit | Under daily cap | Under daily cap |
| No open position | No existing GRX position | No existing GRX position |
| Margin check | Sufficient free margin | Sufficient free margin |
| Spread viability | TP1 ≥ spread × ratio | TP1 ≥ spread × ratio |

**Regime-aligned filter:** BUY signals only fire when the HMM is in Bull state. SELL signals only fire in Bear state. Any Chop state (2 or 3 — see State Labels below) suppresses all signals regardless of XGBoost probability. This ensures the signal direction always agrees with the detected market regime.

**`prob_threshold` and `short_threshold` are tuned independently by Optuna** per broker and timeframe, then loaded automatically at live startup. The search ranges are:

| TF | Broker | BUY threshold range | SELL threshold range |
|----|--------|--------------------|--------------------|
| M5 | `headway_cent` | 0.50 – 0.53 | 0.44 – 0.50 |
| M5 | `standard` | 0.55 – 0.60 | 0.40 – 0.45 |
| M15 | `headway_cent` | 0.50 – 0.58 | 0.42 – 0.50 |
| M15 | `standard` | 0.50 – 0.58 | 0.42 – 0.50 |
| H1 | `headway_cent` | 0.50 – 0.58 | 0.42 – 0.50 |
| H1 | `standard` | 0.50 – 0.58 | 0.42 – 0.50 |

M5 standard uses a stricter floor (0.55–0.60) because standard lots cost proportionally more per trade — the model needs higher conviction before entering to cover the spread.

**Spread viability guard:**

| Broker | Applied on | Minimum ratio |
|--------|-----------|--------------|
| `headway_cent` | M5 only | TP1 ≥ 1.5× spread |
| `standard` | All timeframes | TP1 ≥ 3.0× spread |

### Staged Take-Profits

Each signal opens multiple positions (one per TP level) up to `pos_per_trade`. Positions share the same SL but close at separate TPs as price reaches each target.

**M5 — `pos_per_trade` adapts by account tier:**

| Account | Positions opened | TPs used |
|---------|-----------------|----------|
| ≤ $50 (small) | 2 | TP1 (0.8×) + TP2 (1.5×) |
| > $50 (growth) | 3 | TP1 (0.8×) + TP2 (1.5×) + TP3 (3.0×) |

**TP multipliers by timeframe** (SL = ATR × TF multiplier):

| Regime | TF | TP1 | TP2 (Runner) | TP3 (Growth only) | SL ATR mult |
|--------|----|-----|--------------|-------------------|-------------|
| Bull / Bear | M5 | 0.8× SL | 1.5× SL | 3.0× SL | 1.5× |
| Bull / Bear | M15 | 1.0× SL | 2.0× SL | — | 2.0× |
| Bull / Bear | H1 | 1.0× SL | 2.0× SL | — | 2.0× |
| Chop | any | blocked | blocked | blocked | — |

> **M5 TP3 rationale:** TP3 at 3.0× SL only fills on genuine momentum sessions (e.g. London/NY overlap, news-driven moves). On most bars the Hybrid Scalp Protection exits position 3 before price reaches TP3 — the target acts as a ceiling for strong sessions, not the primary exit.

**Full profit protection chain (M5):**

1. **Profit guard** — SL moves to `entry + 2×spread` once price reaches 70% of TP1 (trade becomes risk-free)
2. **Break-even** — runner SL moves to exact entry when TP1 fills
3. **Fixed scalp target** — each position closed independently when floating P&L reaches **+$4 USD**; does not wait for bar close; re-entry evaluated on next bar
4. **Trailing guard** — once a position's peak P&L reaches **$2 USD**, closes if P&L pulls back to ≤ 50% of peak (e.g. peak $3.00 → closes below $1.50); catches stalling trades and bypasses the 5-minute HMM detection delay
5. **Chop-exit** — all positions closed at market if HMM shifts to Chop mid-trade

> Items 3 and 4 (**Hybrid Scalp Protection**) run every 5 seconds between bar closes. Items 1, 2 and 5 fire at bar close. The combination means position 3 is always protected — either TP3 fills in a strong move, or the trailing guard exits it gracefully when momentum dies.

**Disabling the fixed scalp target (if needed):**
```bash
python main.py --mode demo --tf M5 --broker standard --balance 15 --profit_target 0
```

**Profit protection for H1 / M15:** Only items 1, 2 and 5 apply. Hybrid Scalp Protection is disabled by default on longer timeframes and must be enabled explicitly with `--profit_target N`.

### Deviation (Slippage Tolerance)

| Condition | M5 | M15 / H1 |
|-----------|----|----------|
| Normal | 30 pts | 20 pts |
| High-vol (HMM self-transition < 0.70) | 50 pts | 50 pts |

---

## Volatility Ensemble (Three-Model XGBoost)

Instead of a single XGBoost model, the system trains **three separate models**, each specialising in a different market volatility regime. ATR percentile thresholds are computed on **in-sample data only** (no look-ahead bias):

| Bucket | ATR condition | Description |
|--------|--------------|-------------|
| `low` | `atr_normalized ≤ p33` | Quiet / tight-range market |
| `med` | `p33 < atr_normalized ≤ p66` | Normal trending market |
| `high` | `atr_normalized > p66` | High-volatility / news-driven |

During inference each bar is automatically routed to the correct bucket model based on its live `atr_normalized` value.

### Model Files (Broker + TF Specific)

Each broker and timeframe combination produces entirely separate model files:

```
models/
├── hmm_model_H1_headway_cent.pkl       ← H1 cent HMM
├── hmm_model_H1_standard.pkl           ← H1 standard HMM
├── hmm_model_M15_headway_cent.pkl
├── hmm_model_M5_headway_cent.pkl
├── hmm_model_M5_standard.pkl
├── xgb_ensemble_H1_headway_cent.pkl    ← H1 cent XGBoost ensemble
├── xgb_ensemble_H1_standard.pkl
├── xgb_ensemble_M5_headway_cent.pkl
├── xgb_ensemble_M5_standard.pkl
├── study_headway_cent.db               ← Optuna: cent account trials
├── study_standard.db                   ← Optuna: standard account trials
├── m5_meta_headway_cent.json           ← M5 freshness gate (cent)
└── m5_meta_standard.json               ← M5 freshness gate (standard)
```

Optimizing or training for one broker+TF never affects any other combination.

### Features

| Feature | Description |
|---------|-------------|
| `hmm_state` | GaussianHMM regime label (Bull=0, Bear=1, Chop=2) |
| `rsi_slope` | Rate of change of RSI — momentum direction |
| `atr_normalized` | ATR / Close — normalised volatility |
| `prev_log_return` | Previous bar log return — short-term momentum |
| `usdchf_log_return` | USDCHF log return — intraday USD strength proxy *(optional)* |

---

## Risk Management

Position sizing uses a fixed **1% risk rule** per position:

```
lot_size = (1% × account_balance_USD) / (ATR(14) × SL_multiplier)
```

Minimum lot is always 0.01 (micro-lot). All lots are rounded to 2 decimal places.

### Daily Trade Limits

| Account Balance | TF | Regime | Max Positions/Day | Positions/Signal |
|----------------|----|--------|-------------------|-----------------|
| ≤ $50 | M5 / M15 | Any | 4 | **2** |
| ≤ $50 | H1 | Any | 2 | **2** |
| > $50 | Any | Bull / Bear | 3 signals | **3** |
| > $50 | Any | Chop | 2 signals | **3** |

Each signal opens `pos_per_trade` independent positions with separate TPs. The daily counter tracks **individual positions** — with `pos_per_trade=2` and `max=4`, that gives 2 signals per day on M5/M15 small accounts.

**Lot floor:** All positions are individually floored to **0.01 lots** regardless of what the 1% risk formula calculates. On a $15 account, `lot = (0.01 × 15) / sl_distance` typically computes to exactly 0.01 — the floor is a safety net, not a restriction.

M5 and M15 get a higher cap (4 positions/day) because their higher bar frequency produces more signal opportunities per session. H1 stays at 2 positions/day.

---

## Timeframe Configurations

| Parameter | M5 | M15 | H1 |
|-----------|-----|-----|-----|
| Kalman `obs_cov` default | 0.05 | 4.0 | 1.0 |
| Bars/day (annualization) | 288 | 96 | 24 |
| HMM `n_states` search space | `{2, 4}` only | 3–4 | 3–4 |
| Min OOS trades (optimizer) | 300 | 200 | 200 |
| Positions/signal (≤ $50) | **2** | **2** | **2** |
| Positions/signal (> $50) | **3** | **3** | **3** |
| Max positions/day (≤ $50) | 4 | 4 | 2 |
| SL ATR multiplier | 1.5× | 2.0× | 2.0× |
| TP1 multiplier (trending) | 0.8× SL | 1.0× SL | 1.0× SL |
| TP2 multiplier (runner) | 1.5× SL | 2.0× SL | 2.0× SL |
| TP3 multiplier (growth only) | **3.0× SL** | — | — |
| Chop TP (single, never fires) | 0.5× SL | 0.8× SL | 1.0× SL |
| Profit guard trigger | 70% to TP1 | 70% to TP1 | 70% to TP1 |
| Break-even after TP1 | Yes | Yes | Yes |
| Fixed scalp target | **$4 USD default** | Off | Off |
| Trailing guard activation | **$2 peak** | Off | Off |
| Trailing guard drawdown | **50% of peak** | Off | Off |
| Base deviation | 30 pts | 20 pts | 20 pts |
| Spread viability guard | Yes (cent: M5 only, standard: all) | standard only | standard only |
| 5-day readiness gate | Yes | No | No |

> **M5 `n_states` restriction:** n_states=3 is always degenerate for M5 (Bull and Chop collapse to identical means, producing 500K+ HMM transitions). The optimizer searches only `{2, 4}` for M5.
>
> **H1/M15 `n_states` restriction:** n_states=2 is banned when the regime-aligned filter is active — with only Bull and Bear states, every bar gets assigned to one of the two signal-generating states, creating excessive counter-trend signals. Minimum is 3.

**M5 readiness gate:** `models/m5_meta_<broker>.json` must exist and be less than 120 hours (5 days) old. The optimizer writes this file automatically after completing an M5 study. If stale, `--mode demo/live --tf M5` will exit with an error.

### Per-Timeframe Workflows

**H1 — Headway Cent:**
```bash
python main.py --mode process       --tf H1
python main.py --mode optimize      --tf H1 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf H1 --broker headway_cent --balance 15
python main.py --mode report        --tf H1 --broker headway_cent --balance 15
python main.py --mode export        --tf H1 --broker headway_cent
python main.py --mode sync_validate --tf H1 --broker headway_cent --balance 15 --period 6m
python main.py --mode demo          --tf H1 --broker headway_cent --balance 15
python main.py --mode live          --tf H1 --broker headway_cent --balance 15
```

**M15 — Headway Cent:**
```bash
python main.py --mode process       --tf M15
python main.py --mode optimize      --tf M15 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf M15 --broker headway_cent --balance 15
python main.py --mode report        --tf M15 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M15 --broker headway_cent --balance 15 --period 3m
python main.py --mode demo          --tf M15 --broker headway_cent --balance 15
python main.py --mode live          --tf M15 --broker headway_cent --balance 15
```

**M5 — Headway Cent:**
```bash
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf M5 --broker headway_cent --balance 15
python main.py --mode report        --tf M5 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M5 --broker headway_cent --balance 15 --period 3m
python main.py --mode demo          --tf M5 --broker headway_cent --balance 15
python main.py --mode live          --tf M5 --broker headway_cent --balance 15
```

**M5 — Standard account:**
```bash
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker standard --balance 15 --trials 300
python main.py --mode train         --tf M5 --broker standard --balance 15
python main.py --mode report        --tf M5 --broker standard --balance 15
python main.py --mode sync_validate --tf M5 --broker standard --balance 15 --period 3m
python main.py --mode demo          --tf M5 --broker standard --balance 15
python main.py --mode live          --tf M5 --broker standard --balance 15
```

> Standard M5 optimization uses `prob_threshold` range 0.55–0.60 (vs 0.50–0.53 for cent) because each trade costs proportionally more with standard spreads. The stricter threshold ensures the model only enters when conviction is sufficiently high to overcome the spread cost.

---

## Telegram Remote Control

Start the listener alongside your trading session:

```bash
python main.py --mode listen --broker headway_cent --balance 15
```

A keyboard appears in your Telegram chat:

| Button | Action |
|--------|--------|
| 🚀 START TRADING | Launches `--mode live` using `.env` TF/broker/balance defaults |
| 🛑 STOP TRADING | Kills the running live loop process |
| 📉 START OPTIMIZE (M5) | Starts/resumes M5 Optuna study (saves to `study_headway_cent.db`) |
| 📊 BOT STATUS | Last 24h P&L (real USD), win rate, and floating positions |

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

Prints and sends the last 24h deal summary with P&L in real USD per trade.

### Parallel Optimization

Open multiple terminals and run the same optimize command in each — they share the broker-specific study database safely via SQLite locking:

```bash
# Terminal 1
python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
# Terminal 2 (same command)
python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
```

Each terminal runs an independent Optuna worker. This is more reliable than `--n_jobs` for CPU-bound HMM+XGBoost objectives.

---

## MQL5 EA (Alternative Execution)

`mql5/GoldRegimeX.mq5` is a fully self-contained MT5 Expert Advisor that:
- Loads the exported ONNX model directly inside MT5
- Replicates the same regime → signal → risk logic in MQL5
- Supports both cent and standard accounts via the `IsCentAccount` input
- Uses the same `MAGIC_NUMBER = 123456`

**Do not run the EA and the Python bridge simultaneously** — they share the magic number and will double-count daily trades.

To use the EA:
1. Run `--mode export --tf H1 --broker headway_cent` to generate the ONNX file
2. Copy `mql5/GoldRegimeX.mq5` and the generated `.onnx` file to your MT5 `MQL5/Experts/` folder
3. Compile in MetaEditor (F7)
4. Attach to the XAUUSD chart

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: hmm_model_H1_headway_cent.pkl` | Models not trained for this broker+TF | Run `--mode train --tf H1 --broker headway_cent` |
| `FileNotFoundError: m5_meta_headway_cent.json` | M5 not optimized for this broker | Run `--mode optimize --tf M5 --broker headway_cent` |
| `WARNING: No Optuna study found` | Study DB not found or wrong broker | Run `--mode optimize` with matching `--broker` flag |
| `ERROR: Degenerate HMM` during train | Optuna study missing or stale params | Delete `study_<broker>.db`, re-run optimize, then train |
| Validation FAIL every day | Model OOS ends before MT5 sync window | Export fresh CSV from MT5, re-run full pipeline |
| `Order failed: retcode=10006` | No broker connection | Check MT5 connection indicator (bottom-right) |
| `Order failed: retcode=10015` | Price moved past deviation window | Will retry next bar; elevated deviation auto-applies on high-vol |
| `Insufficient margin` repeated | Account below safe minimum for lot | Top up account or check `--balance` value |
| No signals firing — standard M5 | Default threshold too low (0.50) | Re-optimize with `--broker standard` to use the 0.55–0.60 range |
| Double positions / "Open position — holding" with trades=0 | MQL5 EA running alongside Python bridge (same MAGIC_NUMBER) | Remove GoldRegimeX.mq5 EA from chart — the bridge now logs a `CONFLICT` warning at startup if it detects this |
| Cent P&L logged as $0.00 | Exit deal not yet posted to MT5 history | Bridge waits up to 20 retries (~30s) — this is expected |
| Telegram errors in log | Wrong token or no internet | Check `.env`; regenerate bot token via @BotFather if needed |

**Emergency stop:** Press **Ctrl+C** in the terminal. The loop handles `KeyboardInterrupt` cleanly, logs the shutdown, and disconnects from MT5. All open positions remain open — close them manually from the MT5 Trade tab.

---

## Security Notes

- **Never commit `.env`** — it contains your live Telegram token and is in `.gitignore`
- **`.env.example` contains only placeholders** — fill in real values in `.env` only
- `ALLOWED_USER_ID` is the single security gate for Telegram remote commands
- If credentials are accidentally committed, immediately revoke the bot token via @BotFather `/revoke` and generate a new one
- The listener uses Telegram's long-polling API — no public webhook or open port needed

---

## State Labels

These are hardcoded across all Python modules and the MQL5 EA — do not change:

| Label | Integer | Applies when | Meaning |
|-------|---------|-------------|---------|
| Bull | 0 | Always | Highest mean log-return state — BUY signals eligible |
| Bear | 1 | Always | Lowest mean log-return state — SELL signals eligible |
| Chop | 2 | `n_states = 3` | Middle/sideways state — all signals suppressed |
| Chop_Low | 2 | `n_states = 4` | Second-lowest mean return; slight bearish drift but not a clean Bear — signals suppressed |
| Chop_High | 3 | `n_states = 4` | Second-highest mean return; slight bullish drift but not a clean Bull — signals suppressed |

**Why two Chop states on M5 (`n_states = 4`)?**

With four states the HMM can separate the messy middle into two sub-regimes. The naming is based purely on mean log-return rank:

```
sorted by mean return (low → high):
  [Bear=1]  [Chop_Low=2]  [Chop_High=3]  [Bull=0]
```

Chop_Low typically appears during mild pullbacks or consolidation after a downward move. Chop_High appears during sideways drift after upward momentum. Both suppress signals — the division gives the HMM finer resolution when sorting out ambiguous bars, reducing mislabelling that would otherwise push bars into Bull or Bear incorrectly.

During optimization and training logs, you will see these printed as `Chop_Low` and `Chop_High` in the transition matrix output. In the live log the bar status line shows `state=2` or `state=3`, both labelled `Chop` in the human-readable "No signal" message.

---

## Walk-Forward Analysis & Staleness Gate

### What is Walk-Forward Analysis (WFA)?

A static 80/20 IS/OOS split tells you whether the model generalises from early data to later data — once.  Walk-Forward Analysis rolls that split across the entire dataset to ask: *does the model work consistently across all time periods, or only in a few favourable years?*

```
Full dataset (10 years)
  ├── Window 1:  Train [Y1–Y2]  →  Test [Y2 Q3]
  ├── Window 2:  Train [Y1 Q3–Y2 Q3]  →  Test [Y3 Q1]
  ├── Window N:  Train [...]  →  Test [...]
  └── Aggregate: WFE = mean(OOS Sharpe) / mean(IS Sharpe)
```

**Walk-Forward Efficiency (WFE)** is the key metric:

| WFE | Interpretation |
|-----|----------------|
| ≥ 60% | Robust — safe to go live |
| 50–60% | Acceptable — monitor closely |
| < 50% | Fragile — the model curve-fits specific years |

### Run Walk-Forward Analysis

After training, evaluate stability before going live:

```bash
# H1 (default: 365-day IS, 90-day OOS windows)
python main.py --mode wfa --tf H1 --broker headway_cent --balance 15

# M15 (default: 180-day IS, 60-day OOS windows)
python main.py --mode wfa --tf M15 --broker headway_cent --balance 15

# M5 (default: 90-day IS, 30-day OOS windows)
python main.py --mode wfa --tf M5 --broker headway_cent --balance 15

# Custom windows
python main.py --mode wfa --tf H1 --train_days 180 --test_days 45 --broker headway_cent
```

Sample output:
```
=== Walk-Forward Analysis [H1 / headway_cent] ===
  Windows evaluated : 12
  Mean IS  Sharpe   : +1.847
  Mean OOS Sharpe   : +1.023
  Walk-Forward Eff  : 55.4%  [ROBUST ✅]

  Per-window OOS breakdown:
    2021-03 → 2021-06  OOS=+0.923  trades=18  ✅
    2021-06 → 2021-09  OOS=+1.241  trades=22  ✅
    2021-09 → 2021-12  OOS=-0.112  trades=9   ❌
    ...
```

Results are also sent to Telegram.

### Model Staleness Gate

The `--mode live` and `--mode demo` commands automatically check how old the saved model is before starting.  If the model exceeds the staleness threshold for the given timeframe, the live loop aborts and sends a Telegram alert.

| Timeframe | Max model age | Reason |
|-----------|--------------|--------|
| M5  | 14 days | Microstructure regimes shift weekly |
| M15 | 30 days | Intraday momentum patterns change monthly |
| H1  | 30 days | Swing regimes are more stable but drift over months |

When staleness is detected:

```
⚠️ Market Drift/Staleness detected. Pausing trade loop — [M5] model is 18 days old (limit: 14 days).
Re-optimise before going live:
  python main.py --mode optimize --tf M5 --broker headway_cent --trials 500
  python main.py --mode train    --tf M5 --broker headway_cent
```

The same message is sent to Telegram.

**To bypass the gate** (e.g. for demo testing with an older model):

```bash
python main.py --mode live --tf M5 --broker headway_cent --balance 15 --skip_stale_check
```

### Recommended Maintenance Schedule

```bash
# Weekly (M5): check freshness before every live session
python main.py --mode sync_validate --tf M5 --period 3m --broker headway_cent --balance 15
python main.py --mode wfa           --tf M5 --broker headway_cent --balance 15

# If WFE < 50% or sync_validate fails:
python main.py --mode optimize --tf M5  --broker headway_cent --balance 15 --trials 500
python main.py --mode train    --tf M5  --broker headway_cent --balance 15

# Monthly (H1 / M15): same pattern with longer intervals
python main.py --mode wfa     --tf H1  --broker headway_cent --balance 15
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --trials 250
python main.py --mode train    --tf H1 --broker headway_cent --balance 15
```

