# Gold Regime X

A hybrid machine learning trading system for **XAUUSD (Gold)** that combines Hidden Markov Models for regime detection, XGBoost for signal classification, and a stateful **Regime-Confirmation Signal Engine** that gates live trades on regime persistence, XGBoost probability, synth-VIX Z-score, and dynamic ATR band position. Designed for live execution through MetaTrader 5 on both **Headway Cent** (micro) and **Standard** accounts, with full Telegram remote control and health monitoring.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Account Types — Cent vs Standard](#account-types--cent-vs-standard)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Data Setup](#data-setup)
7. [Configuration](#configuration)
8. [Complete Workflow — Start to Live Trading](#complete-workflow--start-to-live-trading)
9. [Command Reference](#command-reference)
10. [Signal Logic — Regime-Confirmation](#signal-logic--regime-confirmation)
11. [Automatic Data Updates](#automatic-data-updates)
12. [Sensitivity Analysis](#sensitivity-analysis)
13. [Interactive Research Notebook](#interactive-research-notebook)
14. [Risk Management](#risk-management)
15. [Timeframe Configurations](#timeframe-configurations)
16. [Performance Metrics & Scoring](#performance-metrics--scoring)
17. [Optimizer Anti-Overfitting Rules](#optimizer-anti-overfitting-rules)
18. [Telegram Remote Control](#telegram-remote-control)
19. [Multi-TF Live Trading](#multi-tf-live-trading)
20. [MQL5 EA (Alternative Execution)](#mql5-ea-alternative-execution)
21. [Combinatorial Purged CV (CPCV)](#combinatorial-purged-cv-cpcv)
22. [Walk-Forward Analysis & Staleness Gate](#walk-forward-analysis--staleness-gate)
23. [Troubleshooting](#troubleshooting)
24. [Security Notes](#security-notes)

---

## How It Works

```
Raw OHLCV CSV
      │
      ▼
Kalman Filter  →  Smoothed log returns
      │
      ▼
GaussianHMM (k-means prior init, Baum-Welch EM)
      │  Observation features: [kalman_return, volatility, rsi_slope]
      │  All StandardScaler-normalised before fit
      ▼
Regime Labels  Bull=0  Bear=1  Chop=2  [Chop_High=3]  (n_states 3–4, optimised per TF)
      │  Median filter smoothing applied post-prediction
      │    kernel: H1=3 bars, M15=5 bars, M5=5 bars
      │    prevents rapid state oscillation between consecutive bars
      │
      ▼
Session Filter  →  London/NY overlap only  08:00–17:59 UTC
      │  Applied after all rolling indicators are computed
      │  Off-session bars excluded from training and inference
      │
      ▼
GMM Volatility Cluster  →  quiet / normal / volatile label
      │
      ▼
XGBoost Three-Model Volatility Ensemble
      │  Base features (always):
      │    [hmm_0, hmm_1, hmm_2 [, hmm_3]  ← OHE, one column per HMM state
      │     gmm_vol_cluster, rsi_slope,
      │     atr_normalized, prev_log_return]
      │  External features (optional, added when masters exist):
      │    usdchf_log_return, xagusd_log_return,
      │    xtiusd_log_return, synth_vix_zscore*, atr_band_position*
      │  Low / Med / High ATR bucket → separate XGBoost model
      │  * computed from XAUUSD price — no external CSV needed
      │
      ▼
      ▼
SignalEngine — Regime Confirmation
      │  update_regime() tracks consecutive bars in current state
      │  should_enter(): bars ≥ 2, XGBoost prob ≥ TF threshold, P(stay) ≥ TF floor
      │  MR entry: ATR band < 0.10 (buy) or ATR band > 0.90 (sell) in Chop state
      │
      ▼
TCN Confidence Scorer — **removed**
      │    (XGBoost probability gate is the direct signal quality filter)
      │
      ▼
IS / OOS Backtest
      │  Bar-by-bar SignalEngine loop — floating drawdown, Sharpe, Recovery Factor,
      │  Profit Factor, MR attribution, MT5-style equity curve
      │
      ▼
Complex Criterion Score  =  RF_c×0.35 + Sharpe_c×0.35 + (PF−1)×0.20 + Edge_c×0.10
      │  (+return_consistency×0.30 bonus cap for M5/M15)
      │  All terms symmetrically clamped — see Scoring section for details
      │
      ▼
Live Bridge  →  MT5 Market Orders
      │  IOC fill, ATR-based SL, staged TPs, ATR trail
      │  Hybrid Scalp Protection on M5 (between-bar, every 5 s)
      │  TF-specific magic numbers: H1=123456, M15=123457, M5=123458
```

The **Optuna optimizer** searches Kalman parameters, HMM state count, and XGBoost hyperparameters (including L1 and L2 regularisation), scoring every trial on **OOS Complex Criterion only** using **Combinatorial Purged Cross-Validation (CPCV)** with `C(4,2) = 6` combinatorial train/test paths. The dataset is split into 4 chronological blocks; each trial evaluates all 6 path combinations with a TF-specific purge gap (H1: 24 bars, M15: 96 bars, M5: 288 bars) to prevent leakage between adjacent train and test blocks. Stage 1 (`--stage xgb`) uses a fast single IS/OOS hold-out (~5× faster, no CPCV); Stage 2 (`--stage trading`) runs full CPCV warm-started from Stage 1 params. Before each optimization run, `ensure_data_updated()` auto-fetches any missing bars from MT5 so the model always trains on the most recent data.

---

## Architecture

| File | Purpose |
|------|---------|
| `src/processor.py` | Kalman filter, log returns, RSI, ATR, GMM vol cluster, per-TF config; **London/NY session gating** (08:00–17:59 UTC, applied after all rolling indicators) |
| `src/engine_hmm.py` | GaussianHMM with k-means prior init; TF persistence boost; **median-filter state smoothing** (kernel H1=3, M15/M5=5) to prevent rapid state oscillation |
| `src/engine_xgb.py` | XGBoost volatility ensemble; three vol-bucket models (Low/Med/High ATR); **`hmm_state` OHE** (`hmm_0`/`hmm_1`/`hmm_2`[`/hmm_3`]) in `prepare_features`; **`atr_band_position`** feature; `compute_regime_stats()` for metadata; ONNX export |
| `src/signal_engine.py` | **Stateful signal engine** — `SignalEngine`; regime-confirmation entry (persistence + XGBoost prob + synth-VIX Z-score + ATR band); exit on regime reversal, persistence collapse, profit erosion, or max hold |
| `src/sensitivity.py` | Z-Score sensitivity sweep — Bull/Bear cutoffs 1.5–3.0σ; outputs ranked table + `reports/sensitivity_<TF>_<broker>.csv/json` |
| `src/backtester.py` | Bar-by-bar backtest via `SignalEngine` — IS/OOS split, broker costs, floating drawdown, MT5 equity curve, MR attribution |
| `src/optimizer.py` | CPCV optimizer — C(4,2)=6 combinatorial purged paths; Calmar-dominant composite scoring (Calmar×0.45 + Sharpe×0.35 + PF×0.15 + Edge×0.05); `CPCV_N_BLOCKS=4`, `CPCV_K_TEST=2`, `CPCV_PURGE_BARS` per TF; two-stage: `--stage xgb` fast hold-out → `--stage trading` full CPCV; `WFO_PARAMS`/`WFO_PARAMS_FAST` retained for backward-compat (notebook); per-broker SQLite resume; RAM guard; Telegram heartbeat |
| `notebooks/GoldRegimeX_Explorer.ipynb` | Interactive research notebook — equity explorer, WFO window analysis (standard/fast mode), feature/regime explorer, parameter sensitivity with WFO comparison, CV Path Inspector (Section 6) |
| `src/risk_manager.py` | AdaptiveRiskManager, CentConverter, DailyEquityGate, broker cost configs |
| `src/visualizer.py` | 6-chart visual report: regime overlay, equity curve, features, transition matrix, dashboard, MT5 balance/equity |
| `src/mt5_sync.py` | MT5 data downloader; **`ensure_data_updated()`** — auto-fetches missing bars before each optimization run |
| `src/validator.py` | Pre-live validation gate — SignalEngine inference + Sharpe threshold + spread-payoff erosion warning |
| `src/mt5_trader.py` | Live execution loop: bar detection, XGBoost inference, `SignalEngine` regime-confirmation entry/exit, order placement, ATR trailing exits, M5 scalp recycling |
| `src/notifier.py` | Telegram message sender |
| `src/auditor.py` | MT5 deal history report |
| `src/data_updater.py` | **Weekly MT5 data pull** — `WeeklyDataUpdater` appends fresh XAUUSD bars to raw CSVs every Sunday |
| `src/guardian.py` | Multi-TF rolling health monitor — fires Telegram alert when rolling Sharpe drops below threshold |
| `src/remote_control.py` | Telegram long-polling bot for remote commands |
| `src/data_consolidator.py` | USDCHF master file builder |
| `src/monte_carlo.py` | Monte Carlo simulation on trained models — distribution of expected return/drawdown outcomes |
| `main.py` | CLI entry point for all modes |
| `mql5/GoldRegimeX.mq5` | MT5 Expert Advisor with ONNX inference (alternative to Python bridge) |

---

## Account Types — Cent vs Standard

Choose your account type before running any command. The `--broker` flag controls lot sizing, P&L conversion, spread guards, and signal calibration throughout the entire pipeline.

### Headway Cent Account (`--broker headway_cent`)

| Item | Cent Account | Example |
|------|-------------|---------|
| Real deposit | $15 USD | Wired to Headway |
| MT5 balance display | 1 500.00 USC | real USD × 100 |
| Minimum lot | 0.01 | = 0.01 oz gold |
| P&L per $1 gold move at 0.01 lot | 0.01 USC | = **$0.0001 real USD** |
| MT5 history shows `+15.00` | ÷ 100 | = **$0.15 real USD** |
| Bridge balance handling | Divides raw balance by 100 | Pass `--balance 15` (real USD) |
| Spread viability guard (M5) | TP1 ≥ 1.5× spread | |

**Best for:** Learning the system and verifying signals with minimal capital at risk.

### Headway Standard Account (`--broker standard`)

| Item | Standard Account | Example |
|------|-----------------|---------|
| Minimum recommended | $500+ USD | |
| Minimum lot | 0.01 | = 1 oz gold |
| P&L per $1 gold move at 0.01 lot | **$1.00 real USD** | 100× more than cent |
| Bridge balance handling | Uses raw balance directly | Pass `--balance 500` |
| Spread viability guard | TP1 ≥ 3.0× spread | All timeframes |

> **Key difference:** On cent, `+$2.40` in MT5 history = `$0.024` real USD. On standard it is exactly `$2.40`. The bridge always logs P&L in real USD regardless of account type.

---

## Prerequisites

- **Windows** — MetaTrader5 Python package is Windows-only
- **Python 3.11**
- **MetaTrader5 terminal** open and logged into your broker account
- **XAUUSD** visible in Market Watch
- **Algorithmic trading enabled** in MT5: Tools → Options → Expert Advisors → Allow Algorithmic Trading
- A **Telegram bot** (optional but strongly recommended for remote control and health alerts)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/lucasmos/GoldRegime_X.git
cd GoldRegime_X

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
copy .env.example .env
```

---

## Data Setup

Export historical OHLCV CSVs from MetaTrader5 (History Center — F2 → XAUUSD → your TF → Export):

| Timeframe | Expected filename |
|-----------|-------------------|
| H1  | `data/raw/XAU_1h_data.csv` |
| M15 | `data/raw/XAU_15m_data.csv` |
| M5  | `data/raw/XAU_5m_data.csv` |

**Format:** semicolon-delimited, columns: `Date;Open;High;Low;Close;Volume`

> The system filters to the **last 10 years** anchored at the end of your CSV. Export fresh data before each full pipeline re-run.

### Getting deep history from MT5

1. Tools → Options → Charts → Max bars: `99,999,999`
2. View → Symbols (`Ctrl+U`) → XAUUSD → Bars tab → Request (repeat until start year is reached)
3. Export Bars → save as `.csv`

### External Asset Features (optional, recommended)

The processor enriches each XAUUSD bar with cross-asset log returns. Export each asset from MT5 and run `--mode consolidate` to build the master files.

| Asset | Role | Source files (`data/raw/`) | Master produced |
|-------|------|---------------------------|-----------------|
| USDCHF | Intraday DXY proxy (~0.85 correlation) | `USDCHF_H1.csv` / `USDCHF_M15_*.csv` / `USDCHF_M5_*.csv` | `USDCHF_master[_M15/_M5].csv` |
| XAGUSD | Silver — cross-commodity regime signal | `XAGUSD_H1.csv` / `XAGUSD_M15_*.csv` / `XAGUSD_M5_*.csv` | `XAGUSD_master[_M15/_M5].csv` |
| XTIUSD | WTI crude — macro risk-on/off signal | `XTIUSD_H1.csv` / `XTIUSD_M15_*.csv` / `XTIUSD_M5_*.csv` | `XTIUSD_master[_M15/_M5].csv` |
| US500 | S&P 500 — equity/gold correlation | `US500_H1.csv` / `US500_M15_*.csv` / `US500_M5_*.csv` | `US500_master[_M15/_M5].csv` |
| USDJPY | JPY safe-haven proxy | `USDJPY_H1.csv` / `USDJPY_M15_*.csv` / `USDJPY_M5_*.csv` | `USDJPY_master[_M15/_M5].csv` |

All masters live in `data/processed/`. The `synth_vix_zscore` feature (Williams VIX Fix) and `atr_band_position` feature (dynamic ATR band position, 0 = lower band, 1 = upper band) are both computed directly from XAUUSD price — no external file is needed for either.

```bash
python main.py --mode consolidate
```

The pipeline degrades gracefully: any asset whose master file is absent is simply omitted from the feature set. XGBoost's `get_feature_cols()` only includes a column when it is >50% non-null.

---

## Configuration

### 1. Create a Telegram Bot (optional but recommended)

1. Open Telegram → search **@BotFather** → `/newbot`
2. Copy the **Bot Token**
3. Find your **Chat ID** via `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find your **User ID** via [@userinfobot](https://t.me/userinfobot)

### 2. Edit your `.env` file

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ALLOWED_USER_ID=your_telegram_user_id_here

# Defaults used by the Telegram START TRADING button
LIVE_TF=H1
LIVE_BROKER=headway_cent
LIVE_BALANCE=15
```

---

## Complete Workflow — Start to Live Trading

This is the full sequence for a first-time setup on a **Headway Cent account, H1 timeframe**. Adapt TF and broker flags as needed.

### Phase 0 — One-time setup

```bash
# Consolidate USDCHF CSV exports into per-TF master files
python main.py --mode consolidate
```

### Phase 1 — Build and optimise the model

Model optimisation uses a **two-stage workflow** for best results. Stage 1 is a fast XGB-only
exploration (∼5× faster, no CPCV) that saves the best params to `models/stage1_{tf}_{broker}.json`.
Stage 2 runs the full CPCV optimization warm-started from those params. Both stages are safe to
interrupt and resume. Run `--mode train` after both stages to commit the final model to disk.

```bash
# --- Preparation ---
python main.py --mode process --tf H1

# NOTE: ensure_data_updated() runs automatically at the start of every optimize run
# to fetch any bars missing since your last CSV export — MT5 must be connected.

# === STAGE 1: Fast XGB Exploration (no CPCV, ~5x faster) ===
# Explores the hyperparameter landscape with a single IS/OOS split.
# Saves best params to models/stage1_h1_headway_cent.json.
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --stage xgb --trials 60

# === STAGE 2: Full CPCV Optimization (warm-started from Stage 1) ===
# Runs C(4,2)=6 CPCV path evaluation, seeded from Stage 1 params.
# Saves all trials to models/study_headway_cent.db.
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --stage trading --trials 130

# === COMMIT: Train Final Model ===
# Reads best params from study DB, trains GaussianHMM + XGBoost 3-vol-bucket ensemble.
# Saves: hmm_model_H1_headway_cent.pkl + xgb_ensemble_H1_headway_cent.pkl
python main.py --mode train --tf H1 --broker headway_cent --balance 15

# --- Optional post-training checks ---
python main.py --mode sensitivity --tf H1 --broker headway_cent --balance 15
python main.py --mode montecarlo  --tf H1 --broker headway_cent --balance 15
python main.py --mode report      --tf H1 --broker headway_cent --balance 15
python main.py --mode compare     --tf H1,M15,M5 --broker headway_cent --balance 15
python main.py --mode export      --tf H1 --broker headway_cent
```


### Phase 2 — Validate before going live

```bash
# Download live MT5 data and check rolling Sharpe against trained model.
# Use --period 6m for H1 and M15; --period 3m for M5.
# --period 12m available for a deeper H1 look-back.
python main.py --mode sync_validate --tf H1 --broker headway_cent --balance 15 --period 6m
```

| Status | H1 Sharpe | M15 Sharpe | M5 Sharpe | Action |
|--------|-----------|------------|-----------|--------|
| **PASS** | ≥ 0.25 | ≥ 0.50 | ≥ 0.70 | Proceed |
| **WARN** | 0.05–0.25 | 0.20–0.50 | 0.40–0.70 | Proceed with caution or re-optimise |
| **FAIL** | < 0.05 | < 0.20 | < 0.40 | Re-optimise + retrain before going live |

FAIL exits with code 1 and blocks the live script.

### Phase 3 — Walk-Forward Analysis (optional but recommended)

```bash
python main.py --mode wfa --tf H1 --broker headway_cent --balance 15
```

Checks that the model generalises across time periods. WFE ≥ 60% = robust.

### Phase 4 — Demo test

Connect MT5 to a **demo account** and run:

```bash
python main.py --mode demo --tf H1 --broker headway_cent --balance 15
```

This sends **real orders** to MT5 (no simulation). Use it to verify lot sizes, TP/SL placement, and session limits before risking real money.

### Phase 5 — Go live

Connect MT5 to your **live account** and run:

```bash
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```

Type `YES` when prompted. The live loop:

1. Detects each newly completed bar (polls every 5 s)
2. Fetches 200 bars for Kalman / HMM warm-up
3. Runs Kalman → GMM → HMM → XGBoost inference
4. `SignalEngine.update_regime()` + `should_enter()` — checks persistence, XGBoost prob, synth-VIX Z-score, ATR band position
5. Applies session limits (08:00–17:59 UTC), margin check, spread viability guard
6. Places IOC market orders with ATR-based SL and staged TPs (ATR multiples)
7. Logs closed P&L in real USD after every trade
8. Every Sunday: appends fresh bars to raw CSVs (WeeklyDataUpdater)

> **Important:** Remove the GoldRegimeX.mq5 EA from any XAUUSD chart before starting the same-TF Python bridge — they share the same magic number.

### Phase 6 — Ongoing maintenance

```bash
# Weekly (M5) / monthly (M15, H1):
python main.py --mode sync_validate --tf H1 --broker headway_cent --balance 15 --period 6m
python main.py --mode wfa           --tf H1 --broker headway_cent --balance 15

# If sync_validate fails or WFE < 50% — re-run the two-stage optimize + retrain:
# Stage 1: re-run fast XGB exploration (resumes existing study, adds trials on top)
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --stage xgb --trials 60
# Stage 2: re-run full CPCV (warm-started from updated Stage 1 params)
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --stage trading --trials 130
# Commit: retrain final model with new best params
python main.py --mode train    --tf H1 --broker headway_cent --balance 15
```

The [Guardian](#telegram-remote-control) runs a continuous rolling Sharpe health check and fires Telegram alerts if performance degrades.

---

## Command Reference

```
python main.py --mode <MODE> [OPTIONS]
```

### Modes

| Mode | Description |
|------|-------------|
| `consolidate` | Merge `*USDCHF*.csv` files in `data/raw/` into per-TF USDCHF masters |
| `process` | Process raw CSV → features (Kalman, log returns, RSI, ATR, GMM cluster) |
| `optimize` | Run / resume Optuna hyperparameter search (OOS Complex Criterion scoring). Supports `--stage xgb` (fast XGB exploration) → `--stage trading` (full CPCV, warm-started) two-stage workflow |
| `train` | Train HMM + XGBoost; show IS/OOS breakdown |
| `sensitivity` | Z-Score sensitivity sweep (Bull/Bear cutoffs 1.5–3.0) on trained models |
| `compare` | Side-by-side OOS comparison across TFs ranked by Complex Criterion |
| `export` | Export XGBoost ensemble → ONNX for MQL5 EA |
| `report` | Generate 6-chart visual report → `reports/<TF>_<broker>/` |
| `wfa` | Walk-Forward Analysis — per-fold Complex Criterion scores and WFE ratio |
| `sync_validate` | Download live MT5 data + validate model health + cost audit |
| `demo` | Run live execution loop on MT5 demo account (no YES prompt) |
| `live` | Run live execution loop on MT5 live account (requires YES confirmation) |
| `audit` | Generate and Telegram-send daily MT5 deal report |
| `guardian` | Continuous rolling Sharpe health monitor + Telegram alerts |
| `listen` | Start Telegram remote control bot (with nightly 23:55 summary) |
| `montecarlo` | Monte Carlo simulation on trained models — return/drawdown outcome distribution |
| `extract_consensus` | Extract consensus best parameters across Optuna trials |

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tf` | str | `H1` | Timeframe: `H1`, `M15`, `M5` (or comma-separated for `compare`/`guardian`) |
| `--broker` | str | `standard` | `headway_cent` or `standard` |
| `--balance` | float | 15 | Account size in **real USD** |
| `--trials` | int | 250 | Total Optuna trial target (joint mode). Recommended joint: H1=400, M15=600, M5=1000. Two-stage: see `--stage` |
| `--stage` | str | `None` | Two-stage optimize: `xgb` = Stage 1 fast XGB exploration (no CPCV, ∼5× faster, saves `stage1_{tf}_{broker}.json`); `trading` = Stage 2 full CPCV warm-started from Stage 1. Omit to run standard joint optimization |
| `--period` | str | `3m` | Lookback for MT5 sync: `3m`, `6m`, `12m` |
| `--interval` | int | 3600 | Guardian check interval in seconds |
| `--profit_target` | float | 4.0 on M5 | Quick-profit close in USD per position. Pass `0` to disable on M5 |
| `--skip_stale_check` | flag | off | Bypass model-staleness gate on `live`/`demo` |
| `--train_days` | int | TF default | WFA IS window in calendar days (H1=365, M15=180, M5=90) |
| `--test_days` | int | TF default | WFA OOS step in calendar days (H1=90, M15=60, M5=30) |
| `--n_jobs` | int | 1 | Parallel Optuna workers (advanced) |

> **Note:** `--balance` is always in **real USD**. For a Headway Cent account with $15, pass `--balance 15` — the bridge handles the ×100 display conversion internally.

### Per-Timeframe Recommended Workflows

**H1 — Headway Cent (first time):**
```bash
# One-time setup
python main.py --mode consolidate
python main.py --mode process        --tf H1

# Stage 1: Fast XGB exploration (no CPCV) — saves stage1_h1_headway_cent.json
python main.py --mode optimize       --tf H1 --broker headway_cent --balance 15 --stage xgb --trials 60

# Stage 2: Full CPCV optimization (warm-started from Stage 1)
python main.py --mode optimize       --tf H1 --broker headway_cent --balance 15 --stage trading --trials 130

# Commit: Train final model with best params
python main.py --mode train          --tf H1 --broker headway_cent --balance 15

# Optional checks
python main.py --mode sensitivity    --tf H1 --broker headway_cent --balance 15
python main.py --mode montecarlo     --tf H1 --broker headway_cent --balance 15
python main.py --mode report         --tf H1 --broker headway_cent --balance 15

# Validate + go live
python main.py --mode sync_validate  --tf H1 --broker headway_cent --balance 15 --period 6m
python main.py --mode wfa            --tf H1 --broker headway_cent --balance 15
python main.py --mode demo           --tf H1 --broker headway_cent --balance 15
python main.py --mode live           --tf H1 --broker headway_cent --balance 15
```

**M15 — Headway Cent:**
```bash
python main.py --mode process        --tf M15

# Stage 1: Fast XGB exploration — saves stage1_m15_headway_cent.json
python main.py --mode optimize       --tf M15 --broker headway_cent --balance 15 --stage xgb --trials 100

# Stage 2: Full CPCV optimization (warm-started from Stage 1)
python main.py --mode optimize       --tf M15 --broker headway_cent --balance 15 --stage trading --trials 200

# Commit: Train final model with best params
python main.py --mode train          --tf M15 --broker headway_cent --balance 15

# Optional checks
python main.py --mode sensitivity    --tf M15 --broker headway_cent --balance 15
python main.py --mode montecarlo     --tf M15 --broker headway_cent --balance 15
python main.py --mode report         --tf M15 --broker headway_cent --balance 15

# Validate + go live
python main.py --mode sync_validate  --tf M15 --broker headway_cent --balance 15 --period 6m
python main.py --mode wfa            --tf M15 --broker headway_cent --balance 15
python main.py --mode demo           --tf M15 --broker headway_cent --balance 15
python main.py --mode live           --tf M15 --broker headway_cent --balance 15
```

**M5 — Headway Cent:**
```bash
python main.py --mode process        --tf M5

# Stage 1: Fast XGB exploration — saves stage1_m5_headway_cent.json
python main.py --mode optimize       --tf M5 --broker headway_cent --balance 15 --stage xgb --trials 150

# Stage 2: Full CPCV optimization (warm-started from Stage 1)
python main.py --mode optimize       --tf M5 --broker headway_cent --balance 15 --stage trading --trials 300

# Commit: Train final model with best params
python main.py --mode train          --tf M5 --broker headway_cent --balance 15

# Optional checks
python main.py --mode sensitivity    --tf M5 --broker headway_cent --balance 15
python main.py --mode montecarlo     --tf M5 --broker headway_cent --balance 15
python main.py --mode report         --tf M5 --broker headway_cent --balance 15

# Validate + go live
python main.py --mode sync_validate  --tf M5 --broker headway_cent --balance 15 --period 3m
python main.py --mode wfa            --tf M5 --broker headway_cent --balance 15
python main.py --mode demo           --tf M5 --broker headway_cent --balance 15
python main.py --mode live           --tf M5 --broker headway_cent --balance 15
```

---

## Signal Logic — Regime-Confirmation

### Stateful Regime-Confirmation Engine

The live signal gate is the `SignalEngine` class (`src/signal_engine.py`). It is stateful — it tracks how many consecutive bars the current HMM regime has persisted, whether a trade is open, and what regime the trade was entered in.

> **Note on HMM state encoding:** XGBoost receives the HMM regime as one-hot encoded columns (`hmm_0`, `hmm_1`, `hmm_2` for n_states=3; `hmm_0`–`hmm_3` for n_states=4) rather than a single integer. This prevents the model from treating states as ordinal (0 < 1 < 2) and allows each regime to have an independent feature weight.

**Trend entry conditions (all must pass):**

1. **Regime has persisted** — at least 2 consecutive bars in the same regime (Bull or Bear)
2. **synth_vix_zscore** ≥ `MIN_TREND_ZSCORE` for this TF (volatility expansion confirmation)
3. **XGBoost probability** ≥ TF threshold (H1 only; M15/M5 bypass the XGB gate)
4. **ATR band position** within valid range (trend: < 0.80 for BUY, > 0.20 for SELL)
5. **HMM self-transition probability** ≥ TF persistence floor

| TF | XGBoost threshold | Persistence floor | `MIN_TREND_ZSCORE` | `MIN_MR_ZSCORE` |
|----|-------------------|-------------------|--------------------|-----------------|
| H1 | 0.575 | 0.57 | 0.5 | 1.0 |
| M15 | bypass (0.0) | 0.55 | 1.0 | 1.5 |
| M5 | bypass (0.0) | 0.45 | 1.5 | 2.0 |

#### Mean Reversion entries (Chop state ≥ 2)

MR entries require ATR band position at an extreme: ATR band < 0.10 → MR_BUY; ATR band > 0.90 → MR_SELL. MR positions use 75% of the standard lot size. `synth_vix_zscore ≥ MIN_MR_ZSCORE` is also required.

#### Exit conditions

The engine exits an open trade when **any** of the following fires:

| Exit reason | Condition |
|-------------|-----------|
| `regime_reversal` | HMM state changes from the entry regime |
| `persistence_collapse` | P(stay in regime) drops below the TF persistence floor |
| `profit_erosion` | Current P&L < 40% of peak P&L for this trade |
| `max_hold` | Trade open for H1: 24 bars / M15: 32 bars / M5: 48 bars |

#### Trade gate requirements

A trade fires when **all** of the following pass:

| Gate | Requirement |
|------|-------------|
| Regime confirmed | ≥ 2 consecutive bars in regime |
| synth_vix_zscore | ≥ `MIN_TREND_ZSCORE` for this TF |
| XGBoost prob | ≥ TF threshold (H1 only; M15/M5 bypass) |
| ATR band | trend: position < 0.80 (BUY) or > 0.20 (SELL) |
| Persistence | P(stay in state) ≥ TF floor |
| Direction | BUY only in Bull (0); SELL only in Bear (1); Chop → MR only |
| ATR band (MR) | MR_BUY: ATR band < 0.10; MR_SELL: ATR band > 0.90 |
| Session limit | Under daily trade cap for this TF |
| Margin check | Sufficient free margin for the lot size |
| Spread viability | TP1 ≥ spread × ratio (1.5× cent / 3.0× standard) |
| DailyEquityGate | Floating loss < 5% AND day gain < profit-lock threshold |
| Global Guard | Fewer than 4 GRX positions open across all TFs |

### Mean Reversion in Chop — Live Safety Gates

MR signals are gated by the engine's persistence check (P(stay in Chop state) ≥ TF floor), ATR band extremity (< 0.10 / > 0.90), and `synth_vix_zscore ≥ MIN_MR_ZSCORE`, matching the backtester exactly.

### Logic Audit

Every bar that does not fire a trade, the bridge logs a structured reason:

`[LOGIC AUDIT] BULL | state=0  prob=0.510  bars=1  P(stay)=0.82  vix_z=0.87  atr_band=0.44`

### Spread Viability Guard

| Broker | Applied on | Minimum ratio |
|--------|------------|---------------|
| `headway_cent` | M5 only | TP1 ≥ 1.5× spread |
| `standard` | All timeframes | TP1 ≥ 3.0× spread |

### Two-Sided DailyEquityGate

**Loss gate:** Floating equity drops ≥ 5% → gate locks, all GRX positions closed. No new signals for the rest of the day.

**Profit lock:** Day gain reaches TF-specific threshold → no new entries.

| TF | Profit Lock | Loss Gate |
|----|-------------|-----------|
| M5 | **20%** day gain | 5% loss |
| M15 | **10%** day gain | 5% loss |
| H1 | **10%** day gain | 5% loss |

Both gates reset at UTC midnight.

### ATR Trailing Exits

| Phase | Trigger | Action |
|-------|---------|--------|
| Phase 1 (break-even) | Floating P&L ≥ activation amount | SL → entry + 2×spread |
| Phase 2 (trail) | Activated | Trailing SL at ATR × trail multiplier |

**Activation amounts and trail multipliers by TF:**

| TF | Activation P&L | Trail Multiplier |
|----|----------------|-----------------|
| H1 | **$1.50** | 2.5× ATR |
| M15 | **$1.50** | 1.5× ATR |
| M5 | **$1.00** | 1.5× ATR |

### Staged Take-Profits

| Regime | TF | TP1 | TP2 (Runner) | SL ATR mult |
|--------|----|-----|--------------|-------------|
| Bull / Bear | M5 | 0.8× ATR | 1.5× ATR | 1.5× |
| Bull / Bear | M15 | 1.2× ATR | 2.5× ATR | 2.0× |
| Bull / Bear | H1 | 1.5× ATR | 3.0× ATR | 2.0× |
| Chop (MR) | M5 | 0.5× ATR | — | 1.05× |
| Chop (MR) | M15 | 0.8× ATR | — | 1.4× |
| Chop (MR) | H1 | 1.0× ATR | — | 1.4× |

**Full profit protection chain (all TFs):**

1. **Profit guard** — SL → entry + 2×spread when price reaches 70% of TP1 distance
2. **Break-even** — runner SL → entry when TP1 fills
3. **ATR trail** — activates when floating P&L ≥ TF activation amount

**M5-only Hybrid Scalp Protection** (runs every 5 s between bars):

4. **Fixed scalp target** — position closed when floating P&L ≥ $4.00
5. **Trailing guard** — if peak P&L ≥ $2.00, close when P&L falls to ≤ 50% of peak
6. **Chop-exit** — all positions closed at market if HMM shifts to Chop mid-trade
7. **M5 recycle** — after scalp target close, re-entry is allowed on the same bar if regime unchanged and daily cap not hit

## Automatic Data Updates

### Pre-Optimisation Sync (`ensure_data_updated`)

`src/mt5_sync.py` exposes `ensure_data_updated(tf, symbol)`, which is called automatically at the **start of every `--mode optimize` run** for each requested timeframe. It:

1. Reads the last timestamp from the corresponding raw CSV file (`data/raw/XAU_*_data.csv`)
2. Calculates how many bars are missing up to the current time
3. Fetches only the missing bars from MT5 using `copy_rates_range`
4. Appends new rows in semicolon-delimited format, matching the existing CSV structure
5. Skips silently if MT5 is unavailable — optimization continues with existing data

This ensures the model always trains on the most recent bars without requiring a manual data export before each optimization cycle.

```bash
# Sync happens automatically — just run:
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --trials 400
```

### Weekly Data Updater (`WeeklyDataUpdater`)

`src/data_updater.py` runs automatically inside the live trading loop every **Sunday**. It:

1. Fetches the last ~2 weeks of XAUUSD bars from MT5 for each active TF (H1: 336 bars, M15: 1344, M5: 4032)
2. Deduplicates by timestamp and appends only new rows to the raw CSV training files
3. Writes a `data/raw/.last_auto_update` marker

No manual action is required. The updater is a no-op on non-Sunday days and won't re-run within the same week.

### Guardian daily check

When \--mode guardian\ is running it performs a **daily model health check**. If rolling Sharpe degrades or models appear stale, the guardian fires an alert via Telegram.

---

## Sensitivity Analysis

The sensitivity analysis sweeps Bull/Bear Z-Score cutoffs across a range on your **already-trained model** and shows how trades, Sharpe, drawdown, and profit factor change at each level. Note: the live signal path now uses `SignalEngine` — sensitivity results are informational only.

```bash
python main.py --mode sensitivity --tf H1 --broker headway_cent --balance 15
```

**What it tests:** Bull (`Z_CUTOFF_BULL`) and Bear (`Z_CUTOFF_BEAR`) cutoffs from **1.5σ to 3.0σ** in steps of 0.25σ. MR (Chop) cutoffs are held constant throughout.

**Output — printed table:**

```
Z     OOS Trades  Win%    Sharpe  MaxDD%  PF      RF      Return%  Status
1.50  312         52.3    0.81    14.2    1.43    1.92    18.4
1.75  241         55.1    0.97    11.8    1.61    2.34    22.1     ← current
2.00  188         58.2    1.12    10.1    1.79    2.87    26.3     ← BEST
2.25  134         60.4    1.04    9.8     1.74    2.61    24.1
...
```

**Output files** (written to `reports/`):

```
reports/sensitivity_H1_headway_cent.csv
reports/sensitivity_H1_headway_cent.json
```

The JSON includes `current_z`, `best_z`, and `best_sharpe` for automated comparison.

**When to run sensitivity analysis:**
- After training a new model — confirm the hardcoded TF Z cutoff is near-optimal
- When validation Sharpe is borderline — determine whether a lower cutoff would pass

---

## Interactive Research Notebook

`notebooks/GoldRegimeX_Explorer.ipynb` is a fully interactive Jupyter notebook for exploring, tweaking, and diagnosing the system without touching any source files. Run it from the repo root:

```bash
jupyter notebook notebooks/GoldRegimeX_Explorer.ipynb
```

### Sections

| Section | Feature | Typical runtime |
|---------|---------|-----------------|
| **1 — Data & Model Loader** | Select TF, broker, balance, and **WFO mode** (`standard` / `fast`). Loads processed data, best Optuna params, and trained HMM + XGB models. | ~5 s |
| **2 — Equity Curve Explorer** | Interactive equity curve with IS/OOS split line, drawdown panel, and full metric table. Re-runs instantly on balance/broker change. | instant |
| **3 — WFO Window Analysis** | Runs `_run_wfo` using the loaded best params. Plots per-window OOS score bars. Drill into any window's equity curve with the **Window #** slider. Shows **Walk-Forward Efficiency (WFE) interpretation** label. | ~2–3 min |
| **4 — Feature & Regime Explorer** | Price + HMM regime overlay, regime distribution pie, GMM volatility cluster histogram, feature distributions per regime, XGB feature importance. | instant |
| **5 — Parameter Sensitivity** | Sliders for `n_states`, `max_depth`, `learning_rate`, `reg_alpha`, **`reg_lambda`**, and **`min_child_weight`**. **Run Comparison** re-trains and shows a side-by-side metric bar chart. **Run WFO Score Comparison** runs proper rolling WFO (not just a full-dataset backtest) on both baseline and modified configs. | ~30–60 s / ~4–8 min WFO |
| **6 — CV Path Inspector** | Runs either **WFO IS CV** (per-window inner fold scores) or **CPCV** (C(6,2)=15 paths). Displays a per-path boxplot, per-path score bar chart, and a **consistency score** = `n_profitable_paths / total × 100%`. | ~2–5 min |

### WFO Mode (`standard` vs `fast`)

Selected via the **WFO Mode** dropdown in Section 1. Stored in `_CACHE` and used by Sections 3, 5, and 6.

| Mode | IS window | OOS window | Use case |
|------|-----------|------------|----------|
| `standard` | H1: 1 yr / M15: 1 yr / M5: 1 yr | H1: 90d / M15: 90d / M5: 90d | Default; thorough evaluation |
| `fast` | H1: 6 mo / M15: 6 mo / M5: 6 mo | H1: 45d / M15: 45d / M5: 30d | Quicker feedback; suits intraday regime shifts |

### Walk-Forward Efficiency (WFE) Labels

Section 3 automatically labels the WFE ratio after each run:

| WFE | Label |
|-----|-------|
| ≥ 0.80 | ✅ Excellent (OOS ≥ 80% of IS CV performance) |
| ≥ 0.50 | 🟡 Acceptable (OOS ≥ 50% of IS CV performance) |
| ≥ 0.20 | 🟠 Marginal — consider broader regularization |
| < 0.20 | 🔴 Poor — strong overfitting signal |

---

## Risk Management

### Position Sizing — 1% Risk Rule

```
lot_size = (1% × account_balance_USD) / (ATR(14) × SL_multiplier)
```

Minimum lot is always **0.01**. All lots rounded to 2 decimal places.

### Daily Exposure Limits

| Account Balance | TF | Positions per Signal | Max Trades/Day |
|----------------|----|----------------------|----------------|
| ≤ $50 | H1 / M15 | **2** (cent) / **1** (standard) | 2 |
| ≤ $50 | M5 | **2** (cent) / **1** (standard) | 4 |
| > $50 | Any | **3** | 2–3 depending on regime |

**Global Guard:** skip any signal if ≥ 4 GRX positions are open across all TFs.

### Broker Cost Configs

| Broker | Spread fraction | Commission fraction | Total round-trip |
|--------|----------------|--------------------|--------------------|
| `headway_cent` | 0.02% | 0.02% | **0.04%** |
| `standard` | 0.02% | 0.01% | **0.03%** |

---

## Timeframe Configurations

| Parameter | M5 | M15 | H1 |
|-----------|-----|-----|-----|
| Kalman `obs_cov` default | 0.05 | 4.0 | 1.0 |
| Bars/day (annualisation) | 288 | 96 | 24 |
| HMM `n_states` search space | 3–4 | 3–4 | 3–4 |
| HMM persistence gate (training) | ≥ 0.65 | ≥ 0.65 | ≥ 0.65 |
| IS/OOS split | 65% / 35% | 65% / 35% | **70% / 30%** |
| Min OOS trades (hard floor) | **120** | **60** | 20 |
| Min OOS trades (penalty threshold) | 350 | 140 | 25 |
| SL ATR multiplier | 1.5× | 2.0× | 2.0× |
| TP1 multiplier (trend) | 0.8× SL | 1.0× SL | 1.5× SL |
| TP2 multiplier (runner) | 1.5× SL | 2.0× SL | 3.0× SL |
| Hybrid Scalp Protection | **On (5 s)** | Off | Off |
| Fixed scalp target | **$4 USD** | Off | Off |
| Trailing guard activation | **$2 peak** | Off | Off |
| DailyEquityGate loss | 5% | 5% | 5% |
| DailyEquityGate profit lock | **20%** | **10%** | **10%** |
| Model staleness gate | 14 days | 30 days | 30 days |
| M5 optimisation freshness gate | **120 h** | — | — |
| Recommended `--trials` | 1000 | 600 | 400 |

> **`n_states` restriction (all TFs):** All timeframes search `{3, 4}`. n_states=2 is banned on all TFs — with only Bull/Bear states every bar is signal-eligible, creating excessive counter-trend noise. n_states=3 gives Bull/Bear/Chop; n_states=4 adds a `Chop_High` state for volatile chop regimes. The optimal count is selected by Optuna per study.

---

## Performance Metrics & Scoring

### Complex Criterion Score

```
Score = clamp(RF, -5, 5) × 0.35
      + clamp(Sharpe, -3, 3) × 0.35
      + clamp(PF − 1, -2, 2) × 0.20     ← normalised: breakeven = 0
      + clamp(avg_payoff / $0.035, 0, 2) × 0.10
      + [M5/M15 only] return_consistency × 0.30  (cap +0.30)
```

**Score range:** approximately −5 to +5. A score > 0.5 is considered a worthwhile configuration.

| Component | Weight | Clamp | Measures |
|-----------|--------|-------|---------|
| Recovery Factor (RF) | **0.35** | [−5, 5] | Capital preservation — net profit / floating max DD |
| Sharpe Ratio | **0.35** | [−3, 3] | Return smoothness — prevents outlier-driven strategies |
| Profit Factor − 1 (normalised) | **0.20** | [−2, 2] | Trade quality — breakeven = 0, loss-only = −1 |
| Edge / Spread ratio | **0.10** | [0, 2] | avg payoff vs $0.035 spread proxy |
| Return Consistency | +0.30 bonus cap | — | Weekly P&L stability (M5/M15 only) |

### WFO Variance Penalty

The final WFO score aggregates per-window scores with an increased variance penalty:

```
wfo_score = median(window_scores) − 0.20 × std(window_scores)
```

The 0.20 multiplier (up from 0.15) aligns with the wider score range (−5 to +5) to meaningfully penalise inconsistent strategies.

### All Reported Metrics

| Metric | Description |
|--------|-------------|
| **Score** | Complex Criterion (see above) |
| **Sharpe Ratio** | Annualised return / annualised volatility |
| **Recovery Factor** | Net profit / floating max drawdown (capped 20× for display) |
| **Profit Factor** | Gross wins / gross losses (capped 10× for display) |
| **Expected Payoff** | Mean per-trade return × account_size |
| **Max Drawdown** | Peak-to-trough closed-bar drawdown |
| **Floating Max DD** | Intra-bar adverse excursion using bar High/Low |
| **Win Rate** | Fraction of trades that closed profitable |
| **Trade Count** | Total closed trades in the window |
| **Avg Efficiency** | Mean ATR / spread on active-trade bars |
| **Cost Efficiency** | `1 - (total_costs / gross_profit)` |
| **Total Payout** | `total_return × account_size` in broker currency |
| **Return Consistency** | Weekly P&L stability: `1 - std / (std + |mean|)` |
| **mr_trades** | Count of MR trades (Chop-state signals) |
| **mr_win_rate** | Win rate of MR trades |
| **mr_pnl** | Cumulative log-return from MR trades only |

### IS/OOS Split Ratios

| TF | IS | OOS |
|----|----|-----|
| H1 | 70% | 30% |
| M15 | 65% | 35% |
| M5 | 65% | 35% |

---

## Optimizer Anti-Overfitting Rules

| Rule | Detail |
|------|--------|
| OOS-only scoring | All scoring uses OOS data only — IS is never evaluated in the objective |
| Complex Criterion | `RF_c×0.35 + Sharpe_c×0.35 + (PF−1)×0.20 + Edge_c×0.10` — all terms clamped symmetrically |
| Hard trade floor | OOS trades < hard minimum → score **−50.0** (trial discarded) |
| Progressive trade penalty | OOS trades below soft threshold → score × 0.1 |
| Payoff floor | OOS average edge < $0.035 → score × 0.1 |
| Max drawdown gate | OOS floating DD > 20% → score **−50.0** |
| H1 DD guard | OOS floating DD > 5% → score × 0.5 |
| IS/OOS generalisation | If IS Sharpe > 0.1 and OOS/IS Sharpe < 0.35 → score **−50.0** |
| HMM persistence gate | Any self-transition < 0.65 → score **−100.0** |
| M5 activity bonus | OOS trades > 300 → score × 1.2; OOS trades < 150 → score × 0.5 |
| n_states restriction | All TFs: `{3, 4}` |
| No threshold search | SignalEngine thresholds are hardcoded TF constants — not Optuna parameters |
| Per-broker study isolation | `study_headway_cent.db` and `study_standard.db` never interfere |
| CPCV per trial | Every Optuna trial evaluates all C(4,2)=6 combinatorial purged paths — not a single IS/OOS split; `CPCV_MAX_FLOAT_DD=0.20` prunes paths with > 20% floating DD |
| OOS scaler consistency | OOS features are always scaled with the IS-fitted scaler (no data leakage) |
| State alignment | `states_oos` and `states_is_cv` are re-aligned to `df_aligned.index` after NaN drops |
| L2 regularisation | `reg_lambda` is searched for all TFs — XGBoost default of 1.0 is no longer forced |
| Pruner per TF | M5: `HyperbandPruner` (early-stops expensive trials); H1/M15: `MedianPruner(startup=10, warmup=5)` |

```python
# Hard trade floors (trials below these return -50.0 immediately):
MIN_OOS_TRADES_HARD = {"M5": 120, "M15": 60, "H1": 20}

# Soft trade floors (trials below these have score × 0.1):
TF_MIN_OOS_TRADES   = {"M5": 350, "M15": 140, "H1": 25}

# CPCV trial budgets (Stage 2 / joint optimization):
CPCV_TRIALS = {"H1": 80, "M15": 120, "M5": 200}
# Stage-1 trial budgets (fast hold-out, no CPCV):
STAGE1_TRIALS = {"H1": 60, "M15": 100, "M5": 150}
```

### Optuna Search Space per Timeframe

| Parameter | H1 | M15 | M5 |
|-----------|-----|------|-----|
| `learning_rate` | 0.005 – 0.15 | 0.005 – 0.20 | 0.01 – 0.15 |
| `n_estimators` | 50 – 400 (step 50) | 100 – 500 (step 50) | 200 – 600 (step 50) |
| `max_depth` | 3 – 7 | 3 – 7 | 2 – 4 |
| `min_child_weight` | 5 – 100 | 3 – 30 | 5 – 25 |
| `reg_alpha` (L1) | 1e-6 – 0.5 | 1e-6 – 0.1 | 1e-6 – 0.1 |
| `reg_lambda` (L2) | **0.01 – 2.0** | **1e-6 – 0.1** | **1e-6 – 0.1** |
| `subsample` | 0.5 – 0.9 | 0.5 – 0.9 | 0.55 – 0.85 |
| `colsample_bytree` | 0.4 – 0.9 | 0.5 – 1.0 | 0.4 – 0.8 |

> ⚠️ **Delete the study DB** whenever a fundamental parameter changes (e.g. after adding a new feature, changing n_states search space, altering the score function, enabling OHE for `hmm_state`, or adding session gating). Old trials bias the surrogate model.
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
| 🛑 STOP TRADING | Terminates the running live process |
| 📉 START OPTIMIZE (M5) | Starts / resumes M5 Optuna study |
| 📊 BOT STATUS | Last 24 h P&L, win rate, floating positions |

A nightly summary is sent at **23:55 UTC** while the listener runs.

### Guardian — Continuous health monitor

```bash
python main.py --mode guardian --tf M5,M15,H1 --period 3m \
    --interval 3600 --broker headway_cent --balance 15
```

Every hour: validates rolling Sharpe for each TF — fires Telegram alert if below 0.6.

### Audit — On-demand deal report

```bash
python main.py --mode audit --broker headway_cent --balance 15
```

### Parallel Optimisation

```bash
# Open multiple terminals — each one runs an independent Optuna worker:
Terminal 1:  python main.py --mode optimize --tf M5 --broker headway_cent --trials 1000
Terminal 2:  python main.py --mode optimize --tf M5 --broker headway_cent --trials 1000
```

Workers share the same SQLite study safely via locking.

---

## Multi-TF Live Trading

Each timeframe runs as an **independent Python process** with its own magic number:

| TF | Magic Number | Comment format |
|----|-------------|----------------|
| H1 | `123456` | `GRX_H1_TREND_BUY_s0_tp1` |
| M15 | `123457` | `GRX_M15_TREND_SELL_s1_tp2` |
| M5 | `123458` | `GRX_M5_MR_BUY_s2_tp1` |

### Global Exposure Guard

Before any signal, the bridge counts all open GRX positions across all TFs. If **≥ 4 positions** are open, the signal is skipped.

### Starting multiple TFs

```bash
# Terminal 1
python main.py --mode live --tf H1 --broker headway_cent --balance 15

# Terminal 2
python main.py --mode live --tf M15 --broker headway_cent --balance 15

# Terminal 3
python main.py --mode live --tf M5 --broker headway_cent --balance 15
```

At UTC midnight each bridge sends a daily P&L audit to Telegram.

---

## MQL5 EA (Alternative Execution)

`mql5/GoldRegimeX.mq5` is a self-contained MT5 Expert Advisor that loads the ONNX model directly inside MT5 and replicates the same regime → signal → risk logic in MQL5.

**Do not run the EA and the Python bridge for the same TF simultaneously.**

To use the EA:
1. `python main.py --mode export --tf H1 --broker headway_cent`
2. Copy `mql5/GoldRegimeX.mq5` and the `.onnx` file to `MQL5/Experts/`
3. Compile in MetaEditor (F7) and attach to the XAUUSD chart

---

## Combinatorial Purged CV (CPCV)

Both the Optuna optimizer (`--mode optimize`) and the post-training diagnostic (`--mode wfa`) evaluate XGBoost performance using **Combinatorial Purged Cross-Validation (CPCV)** — not rolling Walk-Forward windows. CPCV eliminates serial-correlation leakage by inserting a purge gap at every train/test boundary and exhausts all combinatorial path orderings rather than relying on a single rolling split.

> **Note:** `_run_wfo`, `WFO_PARAMS`, and `WFO_PARAMS_FAST` are retained in `src/optimizer.py` for backward compatibility with the research notebook only. They are not used by the optimizer or `--mode wfa`.

### How it works

```
Full dataset (e.g., 10 years of H1 bars)
  │
  Split into N_BLOCKS = 4 equal chronological blocks
  │
  Generate all C(4,2) = 6 train/test path combinations
  │
  ├── Path 1: Train=[B1,B2]  purge_gap  Test=[B3,B4]
  ├── Path 2: Train=[B1,B3]  purge_gap  Test=[B2,B4]
  ├── Path 3: Train=[B1,B4]  purge_gap  Test=[B2,B3]
  ├── Path 4: Train=[B2,B3]  purge_gap  Test=[B1,B4]
  ├── Path 5: Train=[B2,B4]  purge_gap  Test=[B1,B3]
  └── Path 6: Train=[B3,B4]  purge_gap  Test=[B1,B2]
  │
  For each path:
    1. HMM fitted strictly on purged training blocks (no lookahead)
    2. HMM applied to full dataset → predict_states
    3. StandardScaler fitted on training rows → applied to full dataset
    4. XGBoost ensemble trained on training path (train_ratio=1.0)
    5. Vectorized backtest on test path blocks only
    6. Path score = Calmar-dominant composite (Calmar×0.45 + Sharpe×0.35 + PF×0.15 + Edge×0.05)
    7. Paths pruned if floating_max_drawdown > 20% OR n_trades < hard floor
  │
  cpcv_score = median(path_scores) − 0.20 × std(path_sharpes)
```

### CPCV Parameters

| Parameter | H1 | M15 | M5 |
|-----------|-----|------|-----|
| `CPCV_N_BLOCKS` | 4 | 4 | 4 |
| `CPCV_K_TEST` | 2 | 2 | 2 |
| Paths per trial `C(4,2)` | 6 | 6 | 6 |
| `CPCV_PURGE_BARS` (embargo) | 24 bars | 96 bars | 288 bars |
| `CPCV_MAX_FLOAT_DD` | 20% | 20% | 20% |
| `MIN_TRADES_PER_PATH` | 15 | 60 | 100 |
| Stage-2 / joint trials | 80 | 120 | 200 |
| Stage-1 trials (hold-out) | 60 | 100 | 150 |

### Two-stage optimization flow

| Stage | CLI flag | Method | Speed | Output |
|-------|----------|--------|-------|--------|
| Stage 1 | `--stage xgb` | Fast single IS/OOS hold-out — no CPCV | ~5× faster | `models/stage1_{tf}_{broker}.json` |
| Stage 2 | `--stage trading` | Full CPCV — C(4,2)=6 paths, warm-started from Stage 1 | Full | `models/study_{broker}.db` |
| Joint | *(omit `--stage`)* | Full CPCV from scratch | Full | `models/study_{broker}.db` |

### CPCV score interpretation

| cpcv_score | Interpretation |
|------------|----------------|
| ≥ +1.0 | ✅ Excellent — strong consistent performance across all 6 paths |
| ≥ +0.3 | 🟡 Acceptable — robust enough for live deployment |
| ≥ 0.0 | 🟠 Marginal — consider additional regularisation |
| < 0.0 | 🔴 Poor — inconsistent paths; re-optimise with more trials |

---

## Walk-Forward Analysis & Staleness Gate

### Walk-Forward Analysis

```
Full dataset
  ├── Window 1:  Train [Y1–Y2]       → Test [Y2 Q3]
  ├── Window 2:  Train [Y1 Q3–Y2 Q3] → Test [Y3 Q1]
  └── Aggregate: WFE = mean(OOS Sharpe) / mean(IS Sharpe)
```

| WFE | Interpretation |
|-----|----------------|
| ≥ 60% | Robust — safe to go live |
| 50–60% | Acceptable — monitor closely |
| < 50% | Fragile — model curve-fits specific years |

```bash
python main.py --mode wfa --tf H1 --broker headway_cent --balance 15
```

### Model Staleness Gate

| TF | Max HMM/XGB model age |
|----|-----------------------|
| M5 | 14 days |
| M15 | 30 days |
| H1 | 30 days |

M5 has an additional **5-day optimisation freshness gate**: `m5_meta_<broker>.json` must exist and be < 120 hours old before going live.

**Bypass the gate for demo testing:**
```bash
python main.py --mode demo --tf M5 --broker headway_cent --balance 15 --skip_stale_check
```

### Recommended Maintenance Schedule

```bash
# Weekly (M5):
python main.py --mode sync_validate --tf M5 --period 3m --broker headway_cent --balance 15
python main.py --mode wfa           --tf M5 --broker headway_cent --balance 15
# If needed:
python main.py --mode optimize      --tf M5 --broker headway_cent --balance 15 --trials 1000
python main.py --mode train         --tf M5 --broker headway_cent --balance 15

# Monthly (H1 / M15):
python main.py --mode wfa       --tf H1 --broker headway_cent --balance 15
python main.py --mode optimize  --tf H1 --broker headway_cent --balance 15 --trials 400
python main.py --mode train     --tf H1 --broker headway_cent --balance 15
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: hmm_model_H1_headway_cent.pkl` | Models not trained | `--mode train --tf H1 --broker headway_cent` |
| `FileNotFoundError: m5_meta_headway_cent.json` | M5 not optimised | `--mode optimize --tf M5 --broker headway_cent` |
| `WARNING: No Optuna study found` | Wrong broker or deleted DB | `--mode optimize` with matching `--broker` |
| `ERROR: Degenerate HMM` during train | Stale params | Delete study DB, re-optimise, then train |
| Validation FAIL every day | Model stale or data ends before sync window | Export fresh CSV from MT5, re-run full pipeline |
| No signals firing | XGBoost prob or persistence below thresholds | Check `[LOGIC AUDIT]` logs for `prob=` and `P(stay)=` values; consider re-training |
| No entry after regime change | `bars_in_regime` < 2 | Engine waits for confirmation — normal on first bar of a new regime |
| WFA shows many ❌ folds | Model curve-fits specific years | Loosen regularisation, increase trials, add more training data |
| `Order failed: retcode=10006` | No broker connection | Check MT5 connection indicator |
| `Order failed: retcode=10015` | Price moved past deviation | Will retry next bar; elevated deviation auto-applies on high-vol |
| `[CONFLICT]` warning at startup | GRX positions from EA already open | Stop EA / other process before starting bridge |
| Double positions | MQL5 EA running alongside Python bridge | Remove GoldRegimeX.mq5 EA from chart |
| Telegram errors in log | Wrong token or unconfigured | Check `.env`; regenerate via @BotFather |

**Emergency stop:** Press **Ctrl+C**. Open positions remain open — close them manually from the MT5 Trade tab.

---

## Security Notes

- **Never commit `.env`** — it is in `.gitignore`
- `.env.example` contains only placeholders
- `ALLOWED_USER_ID` is the single security gate for all Telegram commands
- If credentials are accidentally committed, immediately revoke the bot token via @BotFather `/revoke`
- The listener uses Telegram long-polling — no public webhook or open port required

---

## State Labels

Hardcoded across all Python modules and the MQL5 EA:

| Label | Integer | Signal eligibility |
|-------|---------|-------------------|
| Bull | 0 | BUY (trend) |
| Bear | 1 | SELL (trend) |
| Chop | 2 | MR_BUY / MR_SELL only (`n_states = 3`) |
| Chop_Low | 2 | MR_BUY / MR_SELL only (`n_states = 4`) |
| Chop_High | 3 | MR_SELL only with higher cutoff (`n_states = 4`) |

## Model Files Reference

```
models/
├── hmm_model_H1_headway_cent.pkl
├── hmm_model_H1_standard.pkl
├── hmm_model_M15_headway_cent.pkl
├── hmm_model_M5_headway_cent.pkl
├── xgb_ensemble_H1_headway_cent.pkl
├── xgb_ensemble_M15_headway_cent.pkl
├── xgb_ensemble_M5_headway_cent.pkl
├── xgb_model_H1_headway_cent.onnx               ← MQL5 EA uses this
├── study_headway_cent.db               ← Optuna trials (per-broker, never shared)
├── study_standard.db
└── m5_meta_headway_cent.json           ← M5 optimisation freshness gate
```

Without this fix, the notebook's loaded probabilities came from a single global XGBoost model instead of the three vol-bucket models used by the optimizer and live trader, making Section 2 equity curves and Section 3 WFO scores inconsistent with production results.
