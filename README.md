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
13. [Performance Metrics & Scoring](#performance-metrics--scoring)
14. [Optimizer Anti-Overfitting Rules](#optimizer-anti-overfitting-rules)
15. [Telegram Remote Control](#telegram-remote-control)
16. [Multi-TF Live Trading](#multi-tf-live-trading)
17. [MQL5 EA (Alternative Execution)](#mql5-ea-alternative-execution)
18. [Walk-Forward Analysis & Staleness Gate](#walk-forward-analysis--staleness-gate)
19. [Troubleshooting](#troubleshooting)
20. [Security Notes](#security-notes)

---

## How It Works

```
Raw OHLCV CSV
      │
      ▼
Kalman Filter  →  Log Returns (smoothed)
      │
      ▼
GaussianHMM   →  Regime Labels  (Bull=0, Bear=1, Chop=2/3)
               k-means prior initialisation (means, covars, transmat, startprob)
               seeded before Baum-Welch EM — prevents degenerate convergence
               Observation features (3):
                 [kalman_return, volatility, rsi_slope]
               All 3 features StandardScaler-normalised before fit
      │
      ▼
GMM Vol Cluster  →  Volatility regime label (quiet / normal / volatile)
      │
      ▼
XGBoost       →  Three-Model Volatility Ensemble
               Features (V4, 6 total):
                 [hmm_state, gmm_vol_cluster, rsi_slope,
                  atr_normalized, prev_log_return, usdchf_log_return*]
               Low ATR bucket  → XGBoost model (quiet market)
               Med ATR bucket  → XGBoost model (normal market)
               High ATR bucket → XGBoost model (volatile market)
               * optional — requires data/processed/USDCHF_master.csv
      │
      ▼
StandardScaler  →  All continuous features scaled on IS data only (no leakage)
      │
      ▼
Z-Score Signal Evaluation  →  Per-regime probability calibration
               IS mean/std computed per HMM state → stored in model pkl
               Signal fires when prob is N σ from the IS per-state mean:
                 Bull  → BUY  when z ≥ +2.5σ  (H1/default; M5: +2.8σ; M15: +2.3σ)
                 Bear  → SELL when z ≤ −2.5σ  (H1/default; M5: −2.8σ; M15: −2.3σ)
                 Chop  → MR_BUY  when z ≤ −3.0σ / −3.2σ  (H1/M15 / default)
                         MR_SELL when z ≥ +3.5σ / +3.8σ  (H1/M15 / default)
                         M5: MR cutoffs ±3.5σ / ±4.0σ (state 2/3)
               TF-specific cutoffs applied automatically — no manual tuning
      │
      ▼
Backtester    →  IS / OOS split (TF-specific ratio)
               Metrics: Sharpe, Recovery Factor, Profit Factor,
                        Expected Payoff, Efficiency, Cost Efficiency,
                        Floating Max Drawdown
               MT5-style Balance/Equity curve injected into result dict
      │
      ▼
Complex Criterion Score  →  RF×0.4 + PF×0.3 + Sharpe×0.3
      │
      ▼
Live Bridge   →  MT5 Market Orders  (IOC fill, ATR-based SL, staged TPs)
               Signal routing (Z-Score + live safety gates):
                 Bull → Trend BUY    (z ≥ +2.5σ H1 / +2.3σ M15 / +2.8σ M5)
                 Bear → Trend SELL   (z ≤ −2.5σ H1 / −2.3σ M15 / −2.8σ M5)
                 Chop → MR BUY/SELL  (z ≤ ±3.0σ H1/M15 / ±3.5σ M5 + 3 safety gates)
               MR trades: SL = 0.70 × base ATR SL; comment tagged MR vs TREND
               TF-specific magic: H1=123456, M15=123457, M5=123458
               Global guard: skip if ≥ 4 positions open across all TFs
```

The optimizer (Optuna) searches across Kalman parameters, HMM states, and XGBoost hyperparameters, scoring every trial on the **Complex Criterion** (`RF×0.4 + PF×0.3 + Sharpe×0.3`) computed on **out-of-sample data only**. Signal thresholds are no longer part of the search space — they are derived automatically from the IS per-regime probability distribution at training time.

---

## Architecture

| File | Purpose |
|------|---------|
| `src/processor.py` | Kalman filter, log returns, RSI, ATR, GMM vol cluster, per-TF config |
| `src/engine_hmm.py` | GaussianHMM with k-means prior init; TF persistence boost; regime prediction |
| `src/engine_xgb.py` | XGBoost ensemble training; `compute_regime_stats()` for Z-Score calibration; ONNX export |
| `src/signal_evaluator.py` | **Z-Score signal engine** — `evaluate_signal_fast()` (backtester/optimizer) and `evaluate_signal()` (live, with 3 MR safety gates); **TF-specific cutoffs** via `_TF_CUTOFF_OVERRIDES` |
| `src/backtester.py` | Vectorized NumPy backtest — IS/OOS split, Z-Score signals, broker costs, floating drawdown, MT5 equity curve; **MR attribution** (`mr_trades`, `mr_win_rate`, `mr_pnl`) in result dict |
| `src/optimizer.py` | Optuna study — Complex Criterion scoring, per-broker SQLite crash-safe resume, RAM guard, Telegram heartbeat |
| `src/risk_manager.py` | AdaptiveRiskManager, CentConverter, DailyEquityGate, broker cost configs |
| `src/visualizer.py` | **6-chart report**: regime overlay, equity curve, features, transition matrix, dashboard, MT5 balance/equity chart |
| `src/mt5_sync.py` | MT5 data downloader |
| `src/validator.py` | Pre-live validation gate — Z-Score inference + Sharpe threshold + spread-payoff erosion warning |
| `src/mt5_trader.py` | Live execution loop: bar detection, Z-Score signal evaluation with MR gates, order placement, M5 equity lock |
| `src/notifier.py` | Telegram message sender |
| `src/auditor.py` | MT5 deal history report |
| `src/guardian.py` | Multi-TF rolling health monitor |
| `src/remote_control.py` | Telegram long-polling bot for remote commands |
| `src/data_consolidator.py` | USDCHF master file builder |
| `main.py` | CLI entry point for all modes |
| `mql5/GoldRegimeX.mq5` | MT5 Expert Advisor with ONNX inference (alternative to Python bridge) |

---

## Account Types — Cent vs Standard

Choose your account type before running any command. The `--broker` flag controls lot sizing, P&L conversion, spread guards, and signal calibration throughout the entire pipeline.

### Headway Cent Account (`--broker headway_cent`)

A cent account converts your real USD deposit into "cents" displayed 100× larger in the MT5 terminal.

| Item | Cent Account | Example |
|------|-------------|---------|
| Real deposit | $15 USD | Wired to Headway |
| MT5 balance display | 1500.00 USC | (real USD × 100) |
| Minimum lot | 0.01 | = 0.01 oz gold |
| P&L per $1 gold move at 0.01 lot | 0.01 USC | = **$0.0001 real USD** |
| Trade history shows `+15.00` | ÷ 100 | = **$0.15 real USD** |
| Bridge balance handling | Divides raw balance by 100 automatically | Pass `--balance 15` (real USD) |
| Spread viability guard (M5) | TP1 ≥ 1.5× spread | |
| Payout display | `X.XX Cents (X.XXXX USD)` | Human-readable micro-scale |

**Best for:** Learning the system, verifying signals with real broker execution, and starting with minimal capital at risk.

---

### Headway Standard Account (`--broker standard`)

A standard account operates at full contract size in real USD.

| Item | Standard Account | Example |
|------|-----------------|---------|
| Minimum recommended balance | $15+ USD | |
| Minimum lot | 0.01 | = 1 oz gold |
| P&L per $1 gold move at 0.01 lot | **$1.00 real USD** | 100× more than cent |
| Trade history shows `+2.40` | = **$2.40 real USD** | No conversion needed |
| Bridge balance handling | Uses raw balance directly | Pass `--balance 15` |
| Spread viability guard | TP1 ≥ 3.0× spread | Applied on **all timeframes** |
| Payout display | `$X.XXXX USD` | Full dollar precision |

**Best for:** Scaling to meaningful P&L once the strategy is proven on cent.

> **Key practical difference:** On a cent account, `+$2.40` in MT5 history is actually `$0.024` real USD. On a standard account it is exactly `$2.40`. The bridge always logs P&L in **real USD** regardless of account type.

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

### Getting deep history from MT5

- Go to **Tools → Options → Charts**. Change **"Max bars in chart"** to `99,999,999`.
- Go to **View → Symbols** (`Ctrl+U`). Search for the symbol, select the **Bars** tab, choose your timeframe, and click **Request**.
- Keep clicking **"Request"** or scroll back the chart until the desired start year is reached.
- Click **Export Bars** to save as `.csv`.

> **Pro Tip:** If your broker doesn't have 10 years of history, open a free **MetaQuotes-Demo** account inside MT5.

### Keep your data fresh

The processor filters to the **last 10 years anchored at the end of your CSV**. **Export fresh data from MT5 before each full pipeline re-run** to keep the OOS window aligned with the current market.

### USDCHF — cross-asset USD strength feature (optional, recommended)

USDCHF is used as an intraday DXY proxy. Each trading timeframe uses a matching-frequency master file so bars are aligned during feature merging.

| Trading TF | Source file in `data/raw/` | Master produced |
|------------|---------------------------|-----------------|
| H1  | `USDCHF_H1.csv` | `data/processed/USDCHF_master.csv` |
| M15 | `USDCHF_M15_<dates>.csv` | `data/processed/USDCHF_master_M15.csv` |
| M5  | `USDCHF_M5_<dates>.csv`  | `data/processed/USDCHF_master_M5.csv` |

```bash
python main.py --mode consolidate
```

This runs all three consolidations in one pass. TFs without a source file degrade gracefully to the 5-feature model.

---

## Configuration

### 1. Create a Telegram Bot (optional but strongly recommended)

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Copy the **Bot Token**
3. Find your **Chat ID** via `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Get your **User ID** via [@userinfobot](https://t.me/userinfobot)

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

> If you skip Telegram setup, notifications are simply no-ops.

---

## Full Workflow

### Step 0 — Consolidate USDCHF (if you have USDCHF CSV exports)

```bash
python main.py --mode consolidate
```

Builds three per-TF master files from `data/raw/`. Run once before processing.

### Step 1 — Process raw data

```bash
python main.py --mode process --tf H1
```

Applies Kalman filter, computes log returns, RSI, ATR, GMM volatility cluster, and saves a processed parquet file. Repeat for each timeframe you want to trade.

### Step 2 — Optimize hyperparameters

```bash
python main.py --mode optimize --trials 400 --broker headway_cent --balance 15 --tf H1
```

**Recommended trial counts:**

| TF | Recommended `--trials` | Reason |
|----|----------------------|--------|
| H1 | **400** | Fewer bars/day — search space converges faster |
| M15 | **600** | More signal opportunities |
| M5 | **1000** | Largest search space; regime noise requires more exploration |

Runs an Optuna study that searches across:
- Kalman filter parameters (`obs_cov`, `trans_cov`)
- HMM state count (`n_states`: 3–4 for H1/M15; `{2, 4}` for M5)
- XGBoost parameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`)

**Signal thresholds (`prob_threshold`, `short_threshold`) are no longer in the Optuna search space.** They are replaced by the Z-Score architecture — calibrated automatically from the IS per-regime probability distribution at training time.

Each trial is scored on the **OOS Complex Criterion** (`RF×0.4 + PF×0.3 + Sharpe×0.3`). Progress is saved to a per-broker SQLite database — safe to interrupt and resume later:

| Broker | Study database |
|--------|---------------|
| `headway_cent` | `models/study_headway_cent.db` |
| `standard` | `models/study_standard.db` |

**To resume after interruption**, run the exact same command. `--trials N` is a **total target** — if 200 trials already exist and you pass `--trials 300`, it runs 100 more.

**To start completely fresh:**
```bash
del models\study_headway_cent.db
del models\study_standard.db
```

> ⚠️ **All existing studies are stale** after the Z-Score architecture change. Delete study DBs and re-optimize before training new models.

### Step 3 — Train final model

```bash
python main.py --mode train --broker headway_cent --balance 15 --tf H1
```

Trains HMM (with k-means prior initialization) and the three-model XGBoost volatility ensemble using the best optimizer parameters. After training, **computes IS per-regime Z-Score statistics** (`mean` and `std` per HMM state) and stores them in the model pkl. These calibrate all future signal decisions — no manual threshold tuning required.

Prints a detailed IS/OOS breakdown:

```
  [H1 IS] Score: 2.14 | RF: 3.21 | PF: 1.87 | Payoff: $0.0423 | MaxDD: 8.4% (Floating) | WR: 62.3% | Trades: 247
  [H1 OOS] Score: 1.43 | RF: 2.10 | PF: 1.52 | Payoff: $0.0381 | MaxDD: 11.2% (Floating) | WR: 58.7% | Trades: 89
```

Saves broker- and TF-specific model files:

```
models/hmm_model_H1_headway_cent.pkl
models/xgb_ensemble_H1_headway_cent.pkl   ← includes regime_stats for Z-Score
```

Training **aborts** if the HMM is degenerate (any state self-transition < 0.70).

### Step 4 — Compare timeframes (optional)

```bash
python main.py --mode compare --broker headway_cent --balance 15 --tf H1,M15
```

Side-by-side OOS performance ranked by **Complex Criterion Score**. Supports any comma-separated TF combination.

### Step 5 — Export ONNX model (for MQL5 EA)

```bash
python main.py --mode export --tf H1 --broker headway_cent
```

Converts the XGBoost ensemble to ONNX format for the MQL5 EA.

### Step 6 — Generate visual report

```bash
python main.py --mode report --broker headway_cent --balance 15 --tf H1
```

Saves **6 charts** to `reports/H1_headway_cent/`:

1. **Regime overlay** on price
2. **Equity curve** — IS vs OOS with entry markers (Trend BUY/SELL triangles, MR BUY/SELL circles)
3. **Feature analysis** — importance, RSI distributions, 2D regime scatter, feature pie
4. **HMM transition matrix**
5. **Summary dashboard** — key metrics, Profit Attribution table, USD drawdown, best Optuna params
6. **MT5-style Balance/Equity chart** — dark `#131722` background; Balance (blue staircase), Equity (teal continuous), Deposit Load % (filled); IS/OOS amber divider line

Reports are saved to a broker- and TF-specific folder:

```
reports/
├── H1_headway_cent/
├── H1_standard/
├── M15_headway_cent/
├── M5_headway_cent/
└── M5_standard/
```

### Step 7 — Validate before going live

```bash
python main.py --mode sync_validate --period 3m --broker headway_cent --balance 15 --tf H1
```

Downloads the last 3 months of live MT5 data, runs Z-Score inference, and checks recent Sharpe.

| Status | Sharpe | Action |
|--------|--------|--------|
| **PASS** | ≥ 0.8 | Proceed to live |
| **WARN** | 0.5–0.8 | Proceed with caution or re-optimise |
| **FAIL** | < 0.5 | Re-optimize and retrain before going live |

FAIL exits with code 1 and blocks the live script.

> Use `--period 6m` or `--period 8m` for H1 — a 3-month window produces very few H1 trades.

### Step 8 — Test on demo account

```bash
python main.py --mode demo --tf H1 --broker headway_cent --balance 15
```

Sends **real orders to MT5** on whatever account MT5 is currently logged into. Use a demo account to verify that signals, lot sizes, session limits, and TP/SL logic all behave correctly before risking real money.

### Step 9 — Go live

```bash
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```

You will be prompted to type `YES` to confirm. After confirming, the loop:

1. Detects each newly completed bar (polls every 5 seconds)
2. Fetches 200 bars of OHLCV data for Kalman/HMM warm-up
3. Runs Kalman → GMM → HMM → XGBoost inference
4. Evaluates Z-Score signal with live MR safety gates
5. Applies session limits, margin check, and spread viability guard
6. Places IOC market orders with ATR-based SL and staged TPs
7. Logs closed P&L in real USD after every trade
8. **(M5 only)** Activates Trailing Daily Equity Lock if day gain ≥ 20%

**Important:** Remove the GoldRegimeX.mq5 EA from any XAUUSD chart before starting the same-TF Python bridge — they share the same magic number and will conflict.

---

## Command Reference

```
python main.py --mode <MODE> [OPTIONS]
```

| Mode | Description |
|------|-------------|
| `consolidate` | Merge all `*USDCHF*.csv` files in `data/raw/` into per-TF USDCHF masters |
| `process` | Process raw CSV → parquet (Kalman, log returns, RSI, ATR, GMM vol cluster) |
| `optimize` | Run / resume Optuna hyperparameter search (scored on OOS Complex Criterion) |
| `train` | Train HMM + XGBoost; compute Z-Score regime stats; show IS/OOS breakdown |
| `compare` | Side-by-side OOS comparison across TFs (ranked by Complex Criterion Score) |
| `export` | Export XGBoost ensemble → ONNX |
| `report` | Generate **6-chart** visual report saved to `reports/<TF>_<broker>/` |
| `sync_validate` | Download live MT5 data + validate model health + cost audit |
| `wfa` | Walk-Forward Analysis — per-fold Complex Criterion scores and WFE ratio |
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
| `--trials` | — | Total target Optuna trials. Recommended: **M5=1000**, **M15=600**, **H1=400** |
| `--period` | `3m` | Lookback period for sync/validate: `3m`, `6m`, `12m` |
| `--interval` | `3600` | Guardian check interval in seconds |
| `--profit_target` | 4.0 on M5, off elsewhere | Quick-profit close threshold in USD (per position). Pass `0` to disable on M5 |

> **Note:** `--balance` is always in **real USD**. For a Headway Cent account with a $15 deposit, pass `--balance 15` — the bridge handles the ×100 display conversion internally.

> **Removed options:** `--prob_threshold` and `--short_threshold` are no longer used. Signal thresholds are derived automatically from the Z-Score architecture.

---

## Signal Logic

### Z-Score Signal Architecture (V5)

Instead of fixed probability thresholds, signals are calibrated on a **per-regime Z-Score basis**:

1. At training time, `compute_regime_stats()` computes the mean and standard deviation of XGBoost probabilities for IS bars in each HMM state.
2. At inference time, each bar's probability is converted to a Z-Score relative to its HMM state's IS distribution.
3. A signal fires when the Z-Score exceeds the regime-specific cutoff.

**Z-Score cutoffs (TF-specific):**

| State | Signal | H1 cutoff | M15 cutoff | M5 cutoff | High-vol (+0.3σ / +0.4σ M5) |
|-------|--------|-----------|------------|-----------|------------------------------|
| Bull (0) | BUY | z ≥ **+2.5σ** | z ≥ **+2.3σ** | z ≥ **+2.8σ** | +0.3σ added (M5: +0.4σ) |
| Bear (1) | SELL | z ≤ **−2.5σ** | z ≤ **−2.3σ** | z ≤ **−2.8σ** | −0.3σ added (M5: −0.4σ) |
| Chop (2) | MR_BUY | z ≤ **−3.0σ** | z ≤ **−3.0σ** | z ≤ **−3.5σ** | cutoff raised further |
| Chop (2) | MR_SELL | z ≥ **+3.0σ** | z ≥ **+3.0σ** | z ≥ **+3.5σ** | cutoff raised further |
| Chop_High (3, n=4) | MR_SELL | z ≥ **+3.5σ** | z ≥ **+3.5σ** | z ≥ **+4.0σ** | cutoff raised further |

TF-specific cutoffs are applied automatically by `SignalEvaluator(regime_stats, tf=tf)` — no manual configuration needed.

### Trade Gate Requirements

A trade fires when **all** of the following are true:

| Condition | BUY | SELL |
|-----------|-----|------|
| Z-Score | z ≥ +1.5σ in Bull state | z ≤ −1.5σ in Bear state |
| HMM regime  | **Bull state (0) only** | **Bear state (1) only** |
| Chop state | Blocked for trend — MR possible | Blocked for trend — MR possible |
| ER filter | ATR / spread ≥ 1.25 | ATR / spread ≥ 1.25 |
| Session limit | Under daily cap | Under daily cap |
| No open position | No existing GRX position | No existing GRX position |
| Margin check | Sufficient free margin | Sufficient free margin |
| Spread viability | TP1 ≥ spread × ratio | TP1 ≥ spread × ratio |
| DailyEquityGate | Loss < 5% AND day gain < profit lock | Loss < 5% AND day gain < profit lock |
| Global guard | < 4 positions open across all TFs | < 4 positions open across all TFs |

### Mean Reversion in Chop — Three Live Safety Gates

When the HMM is in a Chop state, MR signals are gated by **three additional safety checks** in the live bridge (not applied in backtester/optimizer — these are execution safeguards, not model selection criteria):

| Gate | Requirement | Logic |
|------|-------------|-------|
| **Chop stability** | ≥ 3 consecutive Chop bars | Short-lived Chop transitions are likely noise — wait for confirmed sideways regime |
| **Transition probability** | P(stay in state) ≥ 0.70 from HMM transmat | Low self-transition probability = regime is about to change; avoid fading a potential breakout |
| **Bollinger Band confluence** | MR_BUY: BB position ≤ 0.35; MR_SELL: BB position ≥ 0.65 | Only fade extremes that are also at the outer Bollinger Band (20-period, 2σ) |

All three gates must pass for an MR signal to be placed. A missing gate logs its reason in the structured Logic Audit.

**High-Vol MR Warning:** When `gmm_cluster == 2` (high-volatility), MR signals are gated by raised Z-Score cutoffs (+0.3σ). The bridge also logs `[MR WARNING]` — mean reversion has lower edge during elevated volatility.

### Universal Adaptive Confirmation

After a signal passes Z-Score evaluation, the live bridge applies `_calculate_confirmation_requirements()` to determine whether **additional confirmation bars** are needed before placing an order. Requirements scale with:

- **Signal strength** (Z-Score magnitude) — weaker signals require more bars
- **HMM state** — MR signals require stricter confirmation than trend signals
- **Regime transition context** — re-entering Bull directly from Bear requires more bars than from Chop
- **Volatility cluster** — high-vol bars require lighter confirmation (momentum confirmation)
- **Self-transition probability** — low P(stay) = regime still settling; requires more confirmation

### Logic Audit

Every bar where no trade fires, the bridge logs a structured reason:

- **Low Z-Score** — probability too close to IS mean; weak signal
- **Chop Stability Gate** — fewer than 3 consecutive Chop bars
- **Transition Prob Gate** — P(stay) < 0.70; regime unstable
- **BB Confluence Gate** — price not at Bollinger extreme
- **Directional Confirmation** — waiting for required confirmation bars
- **Chop Suppressed** — Chop state but Z-Score not extreme enough for MR
- **ER Filter** — ATR/spread ratio below minimum
- **Daily Cap** / **Global Guard** / **Equity Gate** — risk management limits

### Spread Viability Guard

| Broker | Applied on | Minimum ratio |
|--------|-----------|--------------|
| `headway_cent` | M5 only | TP1 ≥ 1.5× spread |
| `standard` | All timeframes | TP1 ≥ 3.0× spread |

### Two-Sided DailyEquityGate

**Loss gate (all TFs):** Floating equity drops ≥ 5% below start-of-day baseline → gate locks, all GRX positions closed. No new signals for the rest of the day.

**Profit lock:** Day's floating equity gain reaches TF-specific threshold → no new entries.

| Timeframe | Profit Lock Threshold | Loss Gate |
|-----------|----------------------|-----------|
| M5 | **20%** gain on day | 5% loss |
| M15 | **10%** gain on day | 5% loss |
| H1 | **10%** gain on day | 5% loss |

Both gates reset automatically at UTC midnight.

### Staged Take-Profits

| Regime | TF | TP1 | TP2 (Runner) | TP3 (Growth only) | SL ATR mult |
|--------|----|-----|--------------|-------------------|-------------|
| Bull / Bear | M5 | 0.8× SL | 1.5× SL | 3.0× SL | 1.5× |
| Bull / Bear | M15 | 1.0× SL | 2.0× SL | — | 2.0× |
| Bull / Bear | H1 | 1.5× SL | 3.0× SL | — | 2.0× |
| **Chop (MR)** | M5 | 0.5× SL | — | — | **1.5× × 0.70 = 1.05×** |
| **Chop (MR)** | M15 | 0.8× SL | — | — | **2.0× × 0.70 = 1.40×** |
| **Chop (MR)** | H1 | 1.0× SL | — | — | **2.0× × 0.70 = 1.40×** |

MR trades use **70% of the base ATR SL distance** — tighter stop because mean-reversion has a defined fade target and a breakout quickly invalidates the thesis.

**Full profit protection chain (M5):**

1. **Profit guard** — SL moves to `entry + 2×spread` once price reaches 70% of TP1
2. **Break-even** — runner SL moves to exact entry when TP1 fills
3. **Fixed scalp target** — each position closed when floating P&L reaches **+$4 USD** (between-bar, every 5s)
4. **Trailing guard** — once peak P&L reaches **$2 USD**, closes if P&L pulls back to ≤ 50% of peak
5. **Chop-exit** — all positions closed at market if HMM shifts to Chop mid-trade
6. **Daily equity lock** — no new entries once day gain ≥ 20%

Items 3 and 4 (**Hybrid Scalp Protection**) run every 5 seconds between bar closes.

**Disabling the fixed scalp target:**
```bash
python main.py --mode demo --tf M5 --broker standard --balance 15 --profit_target 0
```

### Deviation (Slippage Tolerance)

| Condition | M5 | M15 / H1 |
|-----------|----|----------|
| Normal | 30 pts | 20 pts |
| High-vol (HMM self-transition < 0.70) | 50 pts | 50 pts |

---

## Volatility Ensemble (Three-Model XGBoost)

Instead of a single XGBoost model, the system trains **three separate models**, each specialising in a different volatility bucket. ATR thresholds are computed on **in-sample data only**:

| Bucket | ATR condition | Description |
|--------|--------------|-------------|
| `low` | `atr_normalized ≤ p33` | Quiet / tight-range market |
| `med` | `p33 < atr_normalized ≤ p66` | Normal trending market |
| `high` | `atr_normalized > p66` | High-volatility / news-driven |

### Features (V4 — 6 total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `hmm_state` | GaussianHMM regime label (Bull=0, Bear=1, Chop=2/3) |
| 2 | `gmm_vol_cluster` | GMM-based volatility regime label |
| 3 | `rsi_slope` | Rate of change of RSI — momentum direction |
| 4 | `atr_normalized` | ATR / Close — normalised volatility |
| 5 | `prev_log_return` | Previous bar log return |
| 6 | `usdchf_log_return` | USDCHF log return — intraday USD strength proxy *(optional)* |

All continuous features (3–6) are scaled by a `StandardScaler` fitted on the **IS portion only**.

### Model Files (Broker + TF Specific)

```
models/
├── hmm_model_H1_headway_cent.pkl
├── hmm_model_H1_standard.pkl
├── hmm_model_M15_headway_cent.pkl
├── hmm_model_M5_headway_cent.pkl
├── hmm_model_M5_standard.pkl
├── xgb_ensemble_H1_headway_cent.pkl    ← includes regime_stats (Z-Score calibration)
├── xgb_ensemble_H1_standard.pkl
├── xgb_ensemble_M5_headway_cent.pkl
├── xgb_ensemble_M5_standard.pkl
├── study_headway_cent.db               ← Optuna: cent account trials
├── study_standard.db                   ← Optuna: standard account trials
├── m5_meta_headway_cent.json           ← M5 freshness gate (cent)
└── m5_meta_standard.json               ← M5 freshness gate (standard)
```

---

## Risk Management

Position sizing uses a fixed **1% risk rule** per position:

```
lot_size = (1% × account_balance_USD) / (ATR(14) × SL_multiplier)
```

Minimum lot is always 0.01. All lots are rounded to 2 decimal places.

### Daily Exposure Limits

`DailyEquityGate` handles stop-trading decisions (see above). `AdaptiveRiskManager` handles lot sizing and positions-per-signal:

| Account Balance | TF | Positions/Signal |
|----------------|----|-----------------|
| ≤ $50 | Any | **2** |
| > $50 | Any | **3** |

Maximum total open positions across all TFs: 4 (Global Guard).

**Lot floor:** All positions individually floored to **0.01 lots**.

---

## Timeframe Configurations

| Parameter | M5 | M15 | H1 |
|-----------|-----|-----|-----|
| Kalman `obs_cov` default | 0.05 | 4.0 | 1.0 |
| Bars/day (annualization) | 288 | 96 | 24 |
| HMM `n_states` search space | `{2, 4}` only | 3–4 | 3–4 |
| HMM persistence gate (training) | ≥ 0.65 | ≥ 0.65 | ≥ 0.65 |
| IS/OOS split | 65% / 35% | 65% / 35% | **70% / 30%** |
| Min OOS trades (penalty threshold) | 400 | 150 | 60 |
| Positions/signal (≤ $50) | **2** | **2** | **2** |
| Positions/signal (> $50) | **3** | **3** | **3** |
| SL ATR multiplier | 1.5× | 2.0× | 2.0× |
| TP1 multiplier (trending) | 0.8× SL | 1.0× SL | 1.5× SL |
| TP2 multiplier (runner) | 1.5× SL | 2.0× SL | 3.0× SL |
| TP3 multiplier (growth only) | **3.0× SL** | — | — |
| Hybrid Scalp Protection | **On (every 5s)** | Off | Off |
| Fixed scalp target | **$4 USD** | Off | Off |
| Trailing guard activation | **$2 peak** | Off | Off |
| DailyEquityGate loss limit | 5% | 5% | 5% |
| DailyEquityGate profit lock | **20%** | **10%** | **10%** |
| 5-day readiness gate | Yes | No | No |

> **M5 `n_states` restriction:** n_states=3 is always degenerate for M5 (Bull/Chop collapse to identical means, producing 500K+ HMM transitions). Optimizer uses `{2, 4}` only.
>
> **H1/M15 `n_states` restriction:** n_states=2 is banned — with only Bull/Bear states every bar generates a signal-eligible regime, creating excessive counter-trend noise. Minimum is 3.

### Per-Timeframe Workflows

**H1 — Headway Cent:**
```bash
python main.py --mode process       --tf H1
python main.py --mode optimize      --tf H1 --broker headway_cent --balance 15 --trials 400
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
python main.py --mode optimize      --tf M15 --broker headway_cent --balance 15 --trials 600
python main.py --mode train         --tf M15 --broker headway_cent --balance 15
python main.py --mode report        --tf M15 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M15 --broker headway_cent --balance 15 --period 3m
python main.py --mode demo          --tf M15 --broker headway_cent --balance 15
python main.py --mode live          --tf M15 --broker headway_cent --balance 15
```

**M5 — Headway Cent:**
```bash
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker headway_cent --balance 15 --trials 1000
python main.py --mode train         --tf M5 --broker headway_cent --balance 15
python main.py --mode report        --tf M5 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M5 --broker headway_cent --balance 15 --period 3m
python main.py --mode demo          --tf M5 --broker headway_cent --balance 15
python main.py --mode live          --tf M5 --broker headway_cent --balance 15
```

---

## Performance Metrics & Scoring

### Complex Criterion Score

```
Score = (Recovery Factor × 0.4) + (Profit Factor × 0.3) + (Sharpe Ratio × 0.3)
```

| Component | Weight | Measures |
|-----------|--------|---------|
| Recovery Factor (RF) | 0.4 | Capital preservation — net return / max floating drawdown (capped at 50×) |
| Profit Factor (PF) | 0.3 | Trade quality — gross wins / gross losses (capped at 10×) |
| Sharpe Ratio | 0.3 | Return smoothness — annualised return per unit of volatility |

### All Reported Metrics

| Metric | Description |
|--------|-------------|
| **Score** | Complex Criterion (`RF×0.4 + PF×0.3 + Sharpe×0.3`) |
| **Sharpe Ratio** | Annualised return / annualised volatility |
| **Recovery Factor** | `min(net_profit / floating_max_drawdown, 50)` |
| **Profit Factor** | `min(gross_wins / gross_losses, 10)` |
| **Expected Payoff** | Mean per-trade return × account_size |
| **Max Drawdown** | Floating intra-bar drawdown using bar High/Low |
| **Win Rate** | Fraction of trades that closed profitable |
| **Trade Count** | Total closed trades in the window |
| **Avg Efficiency** | Mean `ATR / spread` on active-trade bars |
| **Cost Efficiency** | `1 - (total_costs / gross_profit)` |
| **Total Payout** | `total_return × account_size` in broker currency |
| **mr_trades** | Count of MR trades (HMM Chop state signals) |
| **mr_win_rate** | Win rate of MR trades separately from trend |
| **mr_pnl** | Cumulative log-return from MR trades only |

`mr_trades`, `mr_win_rate`, and `mr_pnl` — and their `oos_` prefixed equivalents — are included in the backtester result dict and logged per-trial when the optimizer finds any OOS MR trades.

### IS/OOS Split Ratios

| TF | IS (train) | OOS (test) |
|----|-----------|-----------|
| H1 | 70% | 30% |
| M15 | 65% | 35% |
| M5 | 65% | 35% |

---

## Optimizer Anti-Overfitting Rules

| Rule | Detail |
|------|--------|
| OOS-only scoring | All scoring uses OOS data only |
| Complex Criterion | `RF×0.4+PF×0.3+Sharpe×0.3` — prevents high-Sharpe / deep-DD solutions |
| Progressive trade penalty | OOS trades below TF threshold: score × 0.1 |
| Payoff floor | OOS expected payoff < $0.035: score × 0.1 |
| HMM persistence gate | Any self-transition < **0.65**: trial discarded as degenerate |
| n_states restriction | M5: `{2,4}`; H1/M15: `{3,4}` |
| XGBoost regularization | `reg_alpha` 0.01–1.2, `gamma` 0.01–0.5, `max_depth` 3–6 (H1/M15); M5 deeper L1 |
| No threshold search | `prob_threshold` / `short_threshold` removed from Optuna — thresholds are now Z-Score calibrated from training data, not searched by optimizer |
| Per-broker study isolation | `study_headway_cent.db` and `study_standard.db` are independent |

**Trade penalty thresholds (OOS, per TF):**

| TF | Penalty threshold | Hard minimum |
|----|-----------------|-------------|
| H1 | 60 trades | 5 trades |
| M15 | 150 trades | 5 trades |
| M5 | 400 trades | 5 trades |

> ⚠️ **All existing studies are stale.** The Z-Score architecture change removes `prob_threshold` and `short_threshold` from the Optuna search space. Old trials in the study DB include those parameters in their trial payloads, which biases Optuna's surrogate model. **Delete all study DBs and re-optimize before training new models.**
>
> ```bash
> del models\study_headway_cent.db
> del models\study_standard.db
> ```

---

## Telegram Remote Control

Start the listener alongside your trading session:

```bash
python main.py --mode listen --broker headway_cent --balance 15
```

| Button | Action |
|--------|--------|
| 🚀 START TRADING | Launches `--mode live` using `.env` TF/broker/balance defaults |
| 🛑 STOP TRADING | Kills the running live loop process |
| 📉 START OPTIMIZE (M5) | Starts/resumes M5 Optuna study |
| 📊 BOT STATUS | Last 24h P&L, win rate, and floating positions |

A nightly summary is automatically sent at **23:55 UTC** while the listener runs.

### Guardian — Continuous health monitor

```bash
python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600 --broker headway_cent --balance 15
```

Checks rolling Sharpe for each TF every hour. Sends a Telegram alert if any TF drops below 0.6.

### Audit — On-demand deal report

```bash
python main.py --mode audit --broker headway_cent --balance 15
```

### Parallel Optimization

```bash
# Terminal 1
python main.py --mode optimize --tf M5 --broker headway_cent --trials 1000 --balance 15
# Terminal 2 (same command)
python main.py --mode optimize --tf M5 --broker headway_cent --trials 1000 --balance 15
```

Each terminal runs an independent Optuna worker sharing the broker-specific SQLite study.

---

## Multi-TF Live Trading

Each timeframe runs as an **independent Python process** with its own Magic Number:

| Timeframe | Magic Number | Comment format (examples) |
|-----------|-------------|---------------------------|
| H1 | `123456` | `GRX_H1_TREND_BUY_s0_tp1` / `GRX_H1_MR_BUY_s2_tp1` |
| M15 | `123457` | `GRX_M15_TREND_SELL_s1_tp2` / `GRX_M15_MR_SELL_s3_tp1` |
| M5 | `123458` | `GRX_M5_TREND_BUY_s0_tp1` / `GRX_M5_MR_BUY_s2_tp1` |

### Global Exposure Guard

Before any signal across **any** TF, the bridge counts all open GRX positions. If **≥ 4 positions** are open, the signal is skipped with `[GLOBAL GUARD]` logged.

### Starting Multiple TFs

```bash
# Terminal 1 — H1
python main.py --mode live --tf H1 --broker headway_cent --balance 15

# Terminal 2 — M15
python main.py --mode live --tf M15 --broker headway_cent --balance 15

# Terminal 3 — M5
python main.py --mode live --tf M5 --broker headway_cent --balance 15
```

### Midnight Daily P&L Audit

At UTC midnight each bridge automatically sends a daily P&L audit to Telegram breaking down closed trades by timeframe.

---

## MQL5 EA (Alternative Execution)

`mql5/GoldRegimeX.mq5` is a fully self-contained MT5 Expert Advisor that:
- Loads the exported ONNX model directly inside MT5
- Replicates the same regime → signal → risk logic in MQL5
- Supports both cent and standard accounts via the `IsCentAccount` input
- Uses `MAGIC_NUMBER = 123456` (H1 default)

**Do not run the EA and the Python bridge for the same TF simultaneously.**

To use the EA:
1. Run `--mode export --tf H1 --broker headway_cent` to generate the ONNX file
2. Copy `mql5/GoldRegimeX.mq5` and the `.onnx` file to your MT5 `MQL5/Experts/` folder
3. Compile in MetaEditor (F7) and attach to the XAUUSD chart

---

## Walk-Forward Analysis & Staleness Gate

### What is Walk-Forward Analysis (WFA)?

```
Full dataset (10 years)
  ├── Window 1:  Train [Y1–Y2]  →  Test [Y2 Q3]
  ├── Window 2:  Train [Y1 Q3–Y2 Q3]  →  Test [Y3 Q1]
  └── Aggregate: WFE = mean(OOS Sharpe) / mean(IS Sharpe)
```

| WFE | Interpretation |
|-----|----------------|
| ≥ 60% | Robust — safe to go live |
| 50–60% | Acceptable — monitor closely |
| < 50% | Fragile — model curve-fits specific years |

### Run Walk-Forward Analysis

```bash
# H1 (default: 365-day IS, 90-day OOS windows)
python main.py --mode wfa --tf H1 --broker headway_cent --balance 15

# M15
python main.py --mode wfa --tf M15 --broker headway_cent --balance 15

# M5
python main.py --mode wfa --tf M5 --broker headway_cent --balance 15
```

### Model Staleness Gate

| Timeframe | Max model age |
|-----------|--------------|
| M5  | 14 days |
| M15 | 30 days |
| H1  | 30 days |

**To bypass the gate** (demo testing with an older model):
```bash
python main.py --mode live --tf M5 --broker headway_cent --balance 15 --skip_stale_check
```

### Recommended Maintenance Schedule

```bash
# Weekly (M5):
python main.py --mode sync_validate --tf M5 --period 3m --broker headway_cent --balance 15
python main.py --mode wfa           --tf M5 --broker headway_cent --balance 15

# If WFE < 50% or sync_validate fails:
python main.py --mode optimize --tf M5 --broker headway_cent --balance 15 --trials 1000
python main.py --mode train    --tf M5 --broker headway_cent --balance 15

# Monthly (H1 / M15):
python main.py --mode wfa      --tf H1 --broker headway_cent --balance 15
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --trials 400
python main.py --mode train    --tf H1 --broker headway_cent --balance 15
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: hmm_model_H1_headway_cent.pkl` | Models not trained | Run `--mode train --tf H1 --broker headway_cent` |
| `FileNotFoundError: m5_meta_headway_cent.json` | M5 not optimized | Run `--mode optimize --tf M5 --broker headway_cent` |
| `WARNING: No Optuna study found` | Study DB not found or wrong broker | Run `--mode optimize` with matching `--broker` |
| `ERROR: Degenerate HMM` during train | Stale params | Delete `study_<broker>.db`, re-optimize, then train |
| Validation FAIL every day | OOS ends before MT5 sync window | Export fresh CSV from MT5, re-run full pipeline |
| No signals firing | Z-Score cutoff not reached | Check `regime_stats` in model pkl; re-optimize if needed |
| WFA shows many ❌ folds | Model curve-fits specific years | Loosen regularization, increase trials, add more training data |
| `Order failed: retcode=10006` | No broker connection | Check MT5 connection indicator |
| `Order failed: retcode=10015` | Price moved past deviation | Will retry next bar; elevated deviation auto-applies on high-vol |
| `[CONFLICT]` warning at startup | GRX positions already open from EA | Stop the other process / EA before starting the bridge |
| Double positions | MQL5 EA running alongside Python bridge | Remove GoldRegimeX.mq5 EA from chart |
| Telegram errors in log | Wrong token | Check `.env`; regenerate bot token via @BotFather |
| `too many values to unpack` | Old validator code with outdated return values | Ensure all callers unpack 4 values from `prepare_features()` |

**Emergency stop:** Press **Ctrl+C**. Open positions remain open — close them manually from the MT5 Trade tab.

---

## Security Notes

- **Never commit `.env`** — it is in `.gitignore`
- **`.env.example` contains only placeholders**
- `ALLOWED_USER_ID` is the single security gate for Telegram remote commands
- If credentials are accidentally committed, immediately revoke the bot token via @BotFather `/revoke`
- The listener uses Telegram's long-polling API — no public webhook or open port needed

---

## State Labels

These are hardcoded across all Python modules and the MQL5 EA:

| Label | Integer | Applies when | Meaning |
|-------|---------|-------------|---------|
| Bull | 0 | Always | Highest mean log-return state — BUY signals eligible |
| Bear | 1 | Always | Lowest mean log-return state — SELL signals eligible |
| Chop | 2 | `n_states = 3` | Middle/sideways state — MR signals only |
| Chop_Low | 2 | `n_states = 4` | Second-lowest mean return — MR signals only |
| Chop_High | 3 | `n_states = 4` | Second-highest mean return — MR signals only |

**Why two Chop states on M5 (`n_states = 4`)?**

With four states the HMM can separate the messy middle into two sub-regimes:

```
sorted by mean return (low → high):
  [Bear=1]  [Chop_Low=2]  [Chop_High=3]  [Bull=0]
```

Both suppress trend signals and apply Z-Score MR thresholds independently.
