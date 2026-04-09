# Gold Regime X — Remote Control & Telegram Setup

## Prerequisites

Install the required packages:

```bash
pip install requests python-dotenv psutil schedule
```

---

## Step 1 — Create a Telegram Bot

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Follow the prompts and copy the **Bot Token** (looks like `123456789:AAF-abc...`)
3. Send **`/start`** to your new bot in Telegram
4. Find your **Chat ID** by opening this URL in a browser:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   Look for `"chat": {"id": 123456789}` — that number is your Chat ID.
5. Find your **User ID** (same as Chat ID for private chats, or use [@userinfobot](https://t.me/userinfobot))

---

## Step 2 — Configure .env

```bash
copy .env.example .env
```

Fill in the three required values:

```
TELEGRAM_BOT_TOKEN=123456789:AAF-abc...
TELEGRAM_CHAT_ID=123456789
ALLOWED_USER_ID=123456789
```

Optionally set the live trading defaults (used by the START TRADING button):

```
LIVE_TF=H1
LIVE_BROKER=headway_cent
LIVE_BALANCE=15
```

---

## Step 3 — Start the Listener

On your laptop/PC (while MT5 is running):

```bash
python main.py --mode listen --broker headway_cent --balance 15
```

You will see a confirmation message in your Telegram chat and a keyboard appears.

---

## Telegram Keyboard Buttons

| Button | Action |
|--------|--------|
| 🚀 START TRADING | Launches `--mode live` for the TF/broker/balance set in `.env` |
| 🛑 STOP TRADING | Terminates the running live loop process |
| 📉 START OPTIMIZE (M5) | Starts M5 Optuna study (resumes from `study_headway_cent.db` if interrupted) |
| 📊 BOT STATUS | Sends last 24h trade history with P&L, win rate, and floating positions |

> **Note:** The START TRADING button launches `--mode live` (requires YES confirmation
> when launched manually; the listener passes `--yes` automatically as a subprocess).

---

## Additional Modes

### Guardian — Continuous health monitor

```bash
python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600 --broker headway_cent --balance 15
```

Checks rolling Sharpe for each TF every hour. Sends a Telegram alert if Sharpe drops below 0.6.

### Audit — On-demand performance report

```bash
python main.py --mode audit --broker headway_cent --balance 15
```

Prints and sends the last 24h deal summary to Telegram (P&L in real USD, pips per trade).

---

## Nightly Auto-Report

When `--mode listen` is running, a background thread automatically sends the
daily performance summary to Telegram at **23:55 UTC** every night.

No extra configuration needed — just keep the listener running.

---

## Security Notes

- The `ALLOWED_USER_ID` env var is the single security gate — only your Telegram
  ID can issue commands. Keep your `.env` file private.
- Never commit `.env` to git. It is listed in `.gitignore`.
- The listener uses Telegram's long-polling API (no public webhook needed).
- All commands are launched as separate subprocesses so the bot stays responsive.

---

## Optimization Resume (Failsafe)

The optimizer saves every completed trial to a **per-broker SQLite database**:

| Broker | Study database |
|--------|---------------|
| `headway_cent` | `models/study_headway_cent.db` |
| `standard` | `models/study_standard.db` |

Studies from different brokers never interfere. You can optimize both accounts
simultaneously in separate terminals without data corruption.

If an optimization run is interrupted:

1. Press **Ctrl+C** at any time — progress is saved automatically
2. Re-run the same optimize command:
   ```bash
   python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
   ```
3. You will see: `Failsafe: XX/500 trials already in study. Resuming...`

The `--trials N` argument means **total target trials**, not additional. If 200 trials
already exist and you run `--trials 500`, it will run 300 more to reach 500 total.

### To start a completely fresh study

Delete only the broker-specific database you want to reset:

```bash
# Reset cent account study only (standard study is untouched)
del models\study_headway_cent.db

# Reset standard account study only
del models\study_standard.db
```

> **Important:** Delete the study database whenever you change a fundamental
> parameter (e.g. after adding the regime-aligned filter, or after changing the
> n_states range). Old trials with different assumptions will bias the surrogate model.

### Parallel Optimization (Advanced)

For true multi-process parallelism, open **multiple terminals** and run the same
command in each — they all share the same broker study database safely via SQLite locking:

```
Terminal 1: python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
Terminal 2: python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
Terminal 3: python main.py --mode optimize --tf M5 --broker headway_cent --trials 500 --balance 15
```

Each terminal runs an independent Optuna worker. Coordination is handled via SQLite.
This is more reliable than `--n_jobs` for CPU-bound objectives like HMM + XGBoost.
