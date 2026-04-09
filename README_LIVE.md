# Gold Regime X — Live Trading Guide

This guide explains how to operate the Python MT5 Live Bridge for Gold Regime X.
The bridge connects directly to your running MetaTrader5 terminal, validates the
model against recent market data, and executes orders automatically on XAUUSD.

---

## Account Types — Cent vs Standard

Before running the system, decide which account type you are using. They behave
very differently in terms of P&L, lot sizing, and minimum capital requirements.

### Headway Cent Account (`--broker headway_cent`)

A cent account divides your real USD balance into "cents" displayed 100× larger
in the MT5 terminal.

| Item | Cent Account | Example |
|---|---|---|
| Real deposit | $15 USD | Wired to Headway |
| MT5 balance display | 1500.00 USC | (real × 100) |
| Minimum lot | 0.01 | = 0.01 oz gold |
| P&L per $1 gold move (0.01 lot) | 0.01 USC = $0.0001 real | Very small |
| P&L shown in history (MT5) | In USC | Divide by 100 for real USD |
| `+15.00` shown in MT5 history | = **$0.15 real USD** | |
| Bridge auto-conversion | Yes — divides by 100 internally | Pass `--balance 15` (real USD) |

**When to use Cent:** You are testing the strategy with minimal real-money exposure, or
you are starting out with a small deposit ($15–$50). Losses and gains are 1/100th of what
a standard account would produce on the same lot size. Ideal for validating that signals
work correctly before scaling up.

---

### Headway Standard Account (`--broker standard`)

A standard account operates in real USD at full contract size.

| Item | Standard Account | Example |
|---|---|---|
| Minimum recommended balance | $500+ USD | For safe lot sizing |
| Minimum lot | 0.01 | = 1 oz gold |
| P&L per $1 gold move (0.01 lot) | **$1.00 real USD** | 100× more than cent |
| P&L shown in history (MT5) | Directly in USD | What you see is what you get |
| Bridge auto-conversion | None — balance is already USD | Pass `--balance 500` |
| Spread cost guard | TP1 must be ≥ 3× spread | (vs 1.5× on cent) |

**When to use Standard:** You are trading with real money at full scale and your account
balance is large enough to sustain normal drawdown without margin risk. The system enforces
a stricter spread guard (3× vs 1.5×) because each trade costs proportionally more relative
to the P&L potential.

> **Key difference in practice:** On a cent account, a "profitable" trade showing
> `+$2.40` in MT5 history actually earned you `$0.024` real USD. On a standard account,
> `+$2.40` is exactly $2.40. Always remember the 100× scaling when judging performance
> on cent accounts.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows OS | MetaTrader5 Python package is Windows-only |
| MT5 terminal running | Must be open and logged into your account |
| XAUUSD in Market Watch | Right-click Market Watch → Show All, or search XAUUSD |
| Algorithmic trading enabled | MT5 → Tools → Options → Expert Advisors → Allow Algorithmic Trading |
| Models trained | `models/hmm_model_<TF>_<broker>.pkl` and `models/xgb_ensemble_<TF>_<broker>.pkl` must exist |
| Python package | `pip install MetaTrader5>=5.0.45` |
| **EA removed from chart** | The GoldRegimeX.mq5 EA and the Python bridge share `MAGIC_NUMBER = 123456`. Running both simultaneously will double-count daily trades. Remove the EA before starting the bridge. |

---

## Data Requirement

The system trains on H1 data from your CSV file (`data/raw/XAU_1h_data.csv`).
The processor filters to the **last 10 years** anchored at the **end of the CSV**,
then splits 80% IS / 20% OOS:

```
CSV covers 2004–2025  →  filter to last 10yr  →  ~2016–2025
IS (80%):  ~2016 → ~late 2023
OOS (20%): ~late 2023 → Dec 2025
```

The `sync_validate` command pulls **live MT5 data** (Jan 2026 onward) which is
completely outside the training window. The model will fail validation until you
export fresh data from MT5 :

```bash
# In MT5: History Center → XAUUSD → H1 → Export as semicolon CSV
# Save to:  data/raw/XAU_1h_data.csv   (replace the existing file)
# Then re-run the full pipeline (see Quick-Start Workflow below)
```

---

## Quick-Start Workflow

### Step 1 — Train the models

```bash
python main.py --mode process  --tf H1
python main.py --mode optimize --tf H1 --broker headway_cent --balance 15 --trials 300
python main.py --mode train    --tf H1 --broker headway_cent --balance 15
python main.py --mode export   --tf H1 --broker headway_cent
```

### Step 2 — Validate against recent MT5 data

Run this before going live to confirm the model still works on recent price action.

```bash
python main.py --mode sync_validate --tf H1 --period 3m --broker headway_cent --balance 15
```

| Validation Status | Sharpe Range | Action |
|---|---|---|
| **PASS** | ≥ 0.8 | Proceed to Step 3 |
| **WARN** | 0.5 – 0.8 | Proceed with caution, or re-optimise first |
| **FAIL** | < 0.5 | Re-optimise and re-train before going live |

A FAIL status exits with code 1 and blocks the script from continuing.

### Step 3 — Test on demo account first

Connect MT5 to your **demo** account, then run:

```bash
python main.py --mode demo --tf H1 --broker headway_cent --balance 15
```

Demo mode sends **real orders to MT5** (no simulation). It connects to whatever
account MT5 is currently logged into. Use a demo account to verify that signals,
lot sizes, and session limits behave as expected — then review the trade history
in MT5 before going live.

### Step 4 — Switch to live account

Log MT5 into your live account, then run:

```bash
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```

You will be prompted to type `YES` to confirm. This is the only difference from
`--mode demo` — both send real orders; demo skips the confirmation prompt.

> There is no paper trading / simulation mode. All testing should be done on a demo MT5 account.

---

## Multi-Timeframe Workflows

### M15 Timeframe

```bash
python main.py --mode process       --tf M15
python main.py --mode optimize      --tf M15 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf M15 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M15 --period 3m --broker headway_cent --balance 15
python main.py --mode demo          --tf M15 --broker headway_cent --balance 15
```

### M5 Timeframe — Cent Account

M5 requires a **5-day freshness gate** — optimization must have run within 120 hours.

```bash
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf M5 --broker headway_cent --balance 15
python main.py --mode sync_validate --tf M5 --period 3m --broker headway_cent --balance 15
python main.py --mode demo          --tf M5 --broker headway_cent --balance 15
```

### M5 Timeframe — Standard Account

The standard account uses a more conservative probability range (0.55–0.60 vs 0.50–0.53
for cent) because standard lot spreads are proportionally costlier on a $15–$50 account.

```bash
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker standard --balance 15 --trials 300
python main.py --mode train         --tf M5 --broker standard --balance 15
python main.py --mode sync_validate --tf M5 --period 3m --broker standard --balance 15
python main.py --mode demo          --tf M5 --broker standard --balance 15
```

> **Important:** Cent and standard account models are stored in separate files
> (`xgb_ensemble_M5_headway_cent.pkl` vs `xgb_ensemble_M5_standard.pkl`) and use
> separate Optuna study databases (`study_headway_cent.db` vs `study_standard.db`).
> You can optimise and train both without them interfering with each other.

---

## Probability Thresholds

The signal thresholds control when the system fires a BUY or SELL.
Signals are **regime-aligned**: BUY only when HMM is in Bull state (0),
SELL only when HMM is in Bear state (1). Chop state (2) never generates a signal.

Thresholds are tuned by Optuna per broker and timeframe. The search ranges used:

| Timeframe | Broker | BUY threshold range | SELL threshold range | Reason |
|---|---|---|---|---|
| M5 | `headway_cent` | 0.50 – 0.53 | 0.44 – 0.50 | Live probs cluster below 0.56; tight range keeps signal density high |
| M5 | `standard` | 0.55 – 0.60 | 0.40 – 0.45 | Higher per-trade cost → need stronger conviction before entering |
| M15 | `headway_cent` | 0.50 – 0.58 | 0.42 – 0.50 | Moderate range; regime persistence more lasting on 15-min bars |
| M15 | `standard` | 0.50 – 0.58 | 0.42 – 0.50 | Same as cent; spread guard does the heavy filtering |
| H1 | `headway_cent` | 0.50 – 0.58 | 0.42 – 0.50 | Widest range; fewer bars per day so precision matters more |
| H1 | `standard` | 0.50 – 0.58 | 0.42 – 0.50 | Same as cent; standard on H1 is for larger accounts |

The Optuna-tuned values from your study are loaded automatically at runtime.
You can override them manually if needed:

```bash
python main.py --mode live --tf H1 --broker headway_cent --balance 15 \
  --prob_threshold 0.54 --short_threshold 0.46
```

---

## Risk Management Reference

### Session Limits

| Account Balance | Timeframe | Max Trades/Day | Positions per Signal | Total Positions |
|---|---|---|---|---|
| ≤ $50 USD | H1 / M15 | 2 | 1 | 2 maximum |
| ≤ $50 USD | M5 | 4 | 1 | 4 maximum |
| > $50 USD (Bull/Bear) | Any | 3 | 2 | 6 maximum |
| > $50 USD (Chop) | Any | 2 | 2 | 4 maximum |
| > $50 USD + standard broker | Any | 3 / 2 | **1** | 3 / 2 maximum |

> **Standard account guard:** For standard accounts with balance < $50, `pos_per_trade`
> is forced to 1 regardless of tier. One 0.01 standard lot on XAUUSD has ~$48 notional
> value. Running two simultaneously on a $15 account exceeds safe margin usage.

### Lot Sizing (1% Risk Rule)

```
lot_size = (1% of USD balance) / (ATR(14) × SL_multiplier)
```

| TF | SL Multiplier | Example ($15 balance, ATR=$14.27) |
|---|---|---|
| M5 | 1.5× ATR | 0.15 / 21.41 = 0.007 → **0.01 lot** |
| M15 | 2.0× ATR | 0.15 / 28.54 = 0.005 → **0.01 lot** |
| H1 | 2.0× ATR | 0.15 / 28.54 = 0.005 → **0.01 lot** |

Minimum lot is always 0.01 regardless of formula output.

### Take-Profit Structure (Multi-Stage)

Each signal opens two positions when `pos_per_trade = 2`:

| Position | Role | TP Distance | Behaviour |
|---|---|---|---|
| Position 1 (Partial) | Quick lock-in | TP1: 0.8–1.5× SL (M5), 1.0–2.0× SL (M15), 1.5–3.0× SL (H1) | Closes at TP1 |
| Position 2 (Runner) | Trend-follow | TP2 (see above) | SL moved to entry after TP1 hits |

In Chop state, only a single tighter TP is used.

**Profit guard:** When price reaches 70% of the TP1 distance, the stop-loss on all
open positions is moved to entry + 2×spread, making the trade effectively risk-free
before TP1 triggers.

---

## Model Files (Broker-Specific)

Each broker and timeframe combination produces its own model files and Optuna study.
They never overwrite each other:

```
models/
├── hmm_model_H1_headway_cent.pkl
├── hmm_model_H1_standard.pkl
├── hmm_model_M15_headway_cent.pkl
├── hmm_model_M5_headway_cent.pkl
├── hmm_model_M5_standard.pkl
├── xgb_ensemble_H1_headway_cent.pkl
├── xgb_ensemble_H1_standard.pkl
├── xgb_ensemble_M5_headway_cent.pkl
├── xgb_ensemble_M5_standard.pkl
├── xgb_model_H1_headway_cent.onnx      ← EA uses this
├── study_headway_cent.db               ← Optuna trials (cent)
├── study_standard.db                   ← Optuna trials (standard)
├── m5_meta_headway_cent.json           ← M5 freshness gate
└── m5_meta_standard.json
```

---

## Order Specifications

| Parameter | Value | Notes |
|---|---|---|
| Filling type | IOC (Immediate or Cancel) | Standard for Headway ECN |
| Deviation (normal) | 20 points | $0.20 on XAUUSD |
| Deviation (high-vol) | 50 points | When HMM self-transition probability < 0.70 |
| Magic number | 123456 | Must match GoldRegimeX.mq5 if using EA |
| SL | ATR × TF multiplier | Hard stop on every trade |
| Spread guard (cent) | TP1 ≥ 1.5× spread | M5 only |
| Spread guard (standard) | TP1 ≥ 3.0× spread | All timeframes |

IOC means: if the broker cannot fill within the deviation window, the order is
cancelled entirely. This prevents partial exposure during gold news spikes.

---

## Monitoring Live Trading

### Python Log (real-time)

```powershell
Get-Content logs/goldregimex.log -Wait -Tail 30
```

Key log lines to watch:

```
SIGNAL BUY / SIGNAL SELL   — signal fired, orders being placed
Regime: Bull / Bear / Chop — HMM state at bar close
Order filled               — confirmed by broker
Session limit              — daily cap enforced (no more trades today)
TP1 hit — runner SL moved  — first position closed, runner now risk-free
Profit guard triggered     — SL moved to break-even at 70% to TP1
Closed P&L: +$1.23 [WIN]   — trade result in real USD + pips
Unviable trade: TP1 < Nx spread — signal skipped (spread too wide)
Order failed               — execution error (check MT5 Journal tab)
```

### MT5 Terminal

Every position placed by the bridge has:
- **Comment**: `GRX_BUY_p1of2_s0_p0.73` (direction / position index / HMM state / probability)
- **Magic**: 123456

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: models/hmm_model_H1_headway_cent.pkl` | Models not trained | Run `--mode train --tf H1 --broker headway_cent` |
| `WARNING: No Optuna study found` | Study DB not found | Run `--mode optimize` first |
| Validation FAIL every day | Model stale or data ends before MT5 sync period | Export fresh H1 CSV from MT5, re-run full pipeline |
| `ERROR: Degenerate HMM` at train | Optuna study missing or stale params loaded | Run optimize, then re-run train |
| `Order failed: retcode=10006` | No connection to broker | Check MT5 connection indicator (bottom-right) |
| `Order failed: retcode=10015` | Price moved faster than deviation | Will retry next bar; elevated deviation auto-applies on high-vol |
| `Insufficient margin` repeated | Lot size too large for balance | Top up account or reduce `--balance` value |
| Double positions appearing | EA still on chart | Remove GoldRegimeX.mq5 EA from XAUUSD chart |
| Cent P&L showing as $0.00 | Trade closed within 5s (exit deal not yet posted) | The bridge waits up to 20 retries — give it 30s |
| Standard account: no signals on M5 | prob_threshold too low (0.50 defaults) | Re-optimize with `--broker standard` to get the 0.55–0.60 range |

---

## Emergency Stop

Press **Ctrl+C** in the terminal window. The loop handles `KeyboardInterrupt`
gracefully, logs the shutdown message, and calls `mt5.shutdown()`.

All open positions remain open in MT5 after the script exits. Close them manually
from the Trade tab or by placing an opposing order.

---

## Complete Workflow Examples

### Headway Cent — H1 (first time setup)

```bash
# 1. Export fresh XAUUSD H1 data from MT5 to data/raw/XAU_1h_data.csv

# 2. Build pipeline
python main.py --mode process       --tf H1
python main.py --mode optimize      --tf H1 --broker headway_cent --balance 15 --trials 300
python main.py --mode train         --tf H1 --broker headway_cent --balance 15
python main.py --mode export        --tf H1 --broker headway_cent
python main.py --mode report        --tf H1 --broker headway_cent --balance 15

# 3. Validate
python main.py --mode sync_validate --tf H1 --broker headway_cent --balance 15 --period 3m
#   PASS → continue | FAIL → re-optimize + re-train

# 4. Test on demo MT5 account
python main.py --mode demo --tf H1 --broker headway_cent --balance 15

# 5. Go live
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```

### Headway Standard — M5 (fresh setup)

```bash
# 1. Export fresh XAUUSD M5 data from MT5 to data/raw/XAU_5m_data.csv

# 2. Build pipeline (broker=standard uses separate model files and study DB)
python main.py --mode process       --tf M5
python main.py --mode optimize      --tf M5 --broker standard --balance 15 --trials 300
python main.py --mode train         --tf M5 --broker standard --balance 15
python main.py --mode export        --tf M5 --broker standard
python main.py --mode report        --tf M5 --broker standard --balance 15

# 3. Validate
python main.py --mode sync_validate --tf M5 --broker standard --balance 15 --period 3m

# 4. Test on demo MT5 account
python main.py --mode demo --tf M5 --broker standard --balance 15

# 5. Go live
python main.py --mode live --tf M5 --broker standard --balance 15
```

### Daily Routine (any TF/broker)

```bash
# Before market open:
python main.py --mode sync_validate --tf H1 --broker headway_cent --balance 15 --period 3m
#   PASS → start the loop
#   FAIL → run optimize + train, then re-validate

# Start live:
python main.py --mode live --tf H1 --broker headway_cent --balance 15
```
