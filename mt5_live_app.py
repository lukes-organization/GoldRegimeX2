#!/usr/bin/env python3
"""GoldRegimeX -- MT5 Live Trading Desktop App (Tkinter, standalone).

A thin front-end for the CONSOLIDATED notebook-parity engine. It deliberately
re-implements NO strategy logic:

  * All entries/exits are executed by ``src.mt5_trader._run_loop_inner`` -- the
    1:1 live mirror of the explorer-notebook backtester (guard_factor, leg A/B,
    scale-in leg C, regime-3 MR exit, mode-4 trail/time-stop, AdaptiveRiskManager
    lot sizing). The app just drives it in a background thread.
  * The on-screen "signal" panel calls the SAME ``strategy_backtest.latest_signal``
    the engine uses, so what you see is what the engine acts on.
  * The trade table + daily P&L are read from MT5's own records
    (``history_deals_get`` / ``positions_get``) so they match MetaTrader 5 exactly.

What the app adds on top of the engine:
  1. an MT5 login page (demo OR real account: login / password / server),
  2. live account telemetry,
  3. a broker / timeframe / trade-amount selector,
  4. a live signal panel that says WHY there is no trade when flat,
  5. a live trade table (entry & exit price + time, volume, per-trade P&L),
  6. running realised + floating + total daily P&L,
  7. a daily CSV trade log under ./reports/.

Run:
    python mt5_live_app.py

Prerequisites:
  * MetaTrader 5 terminal installed + ``pip install MetaTrader5`` (Windows).
  * Run ``python main.py --mode optimize`` once so the live model bundle exists.
  * PAPER-TRADE on a DEMO login first and confirm the trade log matches a
    backtest over the same window before using a real account.
"""

import os
import sys
import csv
import queue
import threading
from datetime import datetime, timezone

# Make ``src`` importable regardless of where the app is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox

# ---- Engine seam (imported lazily-safely; resolved at runtime on the trader PC)
_ENGINE_ERR = None
try:
    from src.mt5_trader import (
        _run_loop_inner,
        get_account_telemetry,
        ALL_GRX_MAGICS,
        DEFAULT_SYMBOL,
    )
except Exception as _e:            # pragma: no cover - only fails off-terminal
    _ENGINE_ERR = _e
    _run_loop_inner = None
    get_account_telemetry = None
    ALL_GRX_MAGICS = frozenset({123456, 123457, 123458})
    DEFAULT_SYMBOL = "XAUUSD"

try:
    from src.strategy_backtest import latest_signal
except Exception:
    latest_signal = None

try:
    from src.risk_manager import CENT_MULTIPLIER
except Exception:
    CENT_MULTIPLIER = 100.0


POLL_SECONDS = 3.0          # account + trades refresh cadence
SIGNAL_SECONDS = 20.0       # signal-panel refresh cadence (heavier: re-runs model)


# ---------------------------------------------------------------------------
# Stop shim: wraps the real MetaTrader5 module so a Stop button can break the
# engine's ``while True`` loop cleanly (its body already handles KeyboardInterrupt).
# ---------------------------------------------------------------------------
class _StoppableMT5:
    def __init__(self, real, stop_event):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_stop", stop_event)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_real"), name)
        if callable(attr):
            stop = object.__getattribute__(self, "_stop")
            def _wrapped(*a, **k):
                if stop.is_set():
                    raise KeyboardInterrupt("stop requested by UI")
                return attr(*a, **k)
            return _wrapped
        return attr


# ---------------------------------------------------------------------------
# Read-only helpers for the dashboard
# ---------------------------------------------------------------------------
def _describe_signal(sig):
    """Turn a latest_signal() dict into (headline, explanation) for the UI."""
    if not sig:
        return ("NO DATA",
                "latest_signal() returned nothing. Has the live bundle been built "
                "(run 'python main.py --mode optimize' once)?")
    s = int(sig.get("signal", 0))
    rc = int(sig.get("regime_code", 0))
    pu = float(sig.get("prob_up", 0.0) or 0.0)
    pdn = float(sig.get("prob_down", 0.0) or 0.0)
    thr = float(sig.get("threshold", 0.0) or 0.0)
    if s > 0:
        return ("BUY", "Long signal -- engine opens legs A+B on the new bar "
                       "(subject to the spread & 4-position exposure gates).")
    if s < 0:
        return ("SELL", "Short signal -- engine opens legs A+B on the new bar "
                        "(subject to the spread & 4-position exposure gates).")
    reg = {1: "TREND", 3: "MR / shock"}.get(rc, "non-trend (code %d)" % rc)
    if rc != 1:
        return ("NO TRADE", "Regime is %s. Entries only fire in a TREND regime "
                            "(code 1)." % reg)
    strongest = max(pu, pdn)
    if thr > 0 and strongest < thr:
        return ("NO TRADE", "Trend regime, but model confidence %.2f is below the "
                            "entry threshold %.2f." % (strongest, thr))
    return ("NO TRADE", "Trend regime and confidence OK, but there is no valid "
                        "pullback / entry trigger on this bar.")


def _fetch_trades(mt5, magics, broker):
    """Build today's trade rows + realised/floating P&L from MT5's own records."""
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    cent = float(CENT_MULTIPLIER) if broker == "headway_cent" else 1.0

    open_pos = [p for p in (mt5.positions_get() or []) if getattr(p, "magic", 0) in magics]
    open_ids = {int(getattr(p, "identifier", p.ticket)) for p in open_pos}

    rows = []
    realised = 0.0
    deals = mt5.history_deals_get(start, now) or []
    by_pos = {}
    for d in deals:
        if getattr(d, "magic", 0) not in magics:
            continue
        by_pos.setdefault(int(d.position_id), []).append(d)

    for pid, ds in by_pos.items():
        if pid in open_ids:
            continue                       # still open -> rendered from positions_get below
        ds.sort(key=lambda x: x.time)
        entry = next((x for x in ds if x.entry == 0), ds[0])
        outs = [x for x in ds if x.entry == 1]
        pnl = sum((x.profit + x.commission + x.swap) for x in ds) / cent
        rows.append({
            "pos": pid,
            "side": "LONG" if int(entry.type) == 0 else "SHORT",
            "entry_time": datetime.fromtimestamp(entry.time, tz=timezone.utc).strftime("%H:%M:%S"),
            "entry_price": round(float(entry.price), 2),
            "exit_time": datetime.fromtimestamp(outs[-1].time, tz=timezone.utc).strftime("%H:%M:%S") if outs else "-",
            "exit_price": round(float(outs[-1].price), 2) if outs else "-",
            "volume": round(sum(float(x.volume) for x in ds if x.entry == 0), 2),
            "pnl": round(pnl, 2),
            "status": "closed",
        })
        realised += pnl

    floating = 0.0
    for p in open_pos:
        fp = float(p.profit) / cent
        floating += fp
        rows.append({
            "pos": int(getattr(p, "identifier", p.ticket)),
            "side": "LONG" if int(p.type) == 0 else "SHORT",
            "entry_time": datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%H:%M:%S"),
            "entry_price": round(float(p.price_open), 2),
            "exit_time": "-",
            "exit_price": "-",
            "volume": round(float(p.volume), 2),
            "pnl": round(fp, 2),
            "status": "OPEN",
        })

    rows.sort(key=lambda r: r["entry_time"])
    return rows, realised, floating


def _write_csv(rows, realised, floating, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["position", "side", "entry_time", "entry_price", "exit_time",
                    "exit_price", "volume", "pnl_usd", "status"])
        for r in rows:
            w.writerow([r["pos"], r["side"], r["entry_time"], r["entry_price"],
                        r["exit_time"], r["exit_price"], r["volume"], r["pnl"], r["status"]])
        w.writerow([])
        w.writerow(["realised_pnl_usd", round(realised, 2)])
        w.writerow(["floating_pnl_usd", round(floating, 2)])
        w.writerow(["total_pnl_usd", round(realised + floating, 2)])


# ---------------------------------------------------------------------------
# Background poller: reads account / trades / signal and posts to the UI queue.
# ---------------------------------------------------------------------------
class _Poller(threading.Thread):
    def __init__(self, get_ctx, out_q, stop_event):
        super().__init__(daemon=True)
        self._get = get_ctx
        self._q = out_q
        self._stop = stop_event

    def run(self):
        try:
            import MetaTrader5 as mt5
        except Exception as e:
            self._q.put(("error", "MetaTrader5 import failed: %s" % e))
            return
        sig_every = max(1, int(round(SIGNAL_SECONDS / POLL_SECONDS)))
        i = 0
        while not self._stop.is_set():
            tf, broker = self._get()
            try:
                if get_account_telemetry is not None:
                    self._q.put(("account", get_account_telemetry()))
                else:
                    info = mt5.account_info()
                    if info is not None:
                        self._q.put(("account", info._asdict()))
            except Exception as e:
                self._q.put(("error", "account: %s" % e))
            try:
                rows, realised, floating = _fetch_trades(mt5, ALL_GRX_MAGICS, broker)
                self._q.put(("trades", (rows, realised, floating)))
                try:
                    os.makedirs("reports", exist_ok=True)
                    _write_csv(rows, realised, floating,
                               os.path.join("reports", "live_app_trades_%s.csv"
                                            % datetime.now().strftime("%Y-%m-%d")))
                except Exception:
                    pass
            except Exception as e:
                self._q.put(("error", "trades: %s" % e))
            if i % sig_every == 0 and latest_signal is not None:
                try:
                    self._q.put(("signal", latest_signal(tf)))
                except Exception as e:
                    self._q.put(("error", "signal: %s" % e))
            i += 1
            self._stop.wait(POLL_SECONDS)


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------
class LiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GoldRegimeX -- MT5 Live Trader")
        self.mt5 = None
        self.ui_q = queue.Queue()
        self.poll_stop = None
        self.trade_stop = None
        self.trade_thread = None
        self.tf_var = tk.StringVar(value="H1")
        self.broker_var = tk.StringVar(value="headway_cent")
        self.amount_var = tk.StringVar(value="")
        self._build_login()

    # ---- login page ----
    def _build_login(self):
        self.login_frame = ttk.Frame(self.root, padding=16)
        self.login_frame.grid(sticky="nsew")
        ttk.Label(self.login_frame, text="Connect to MetaTrader 5",
                  font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 12))
        self.path_var = tk.StringVar()
        self.login_var = tk.StringVar()
        self.pw_var = tk.StringVar()
        self.server_var = tk.StringVar()
        fields = [
            ("Terminal path (optional)", self.path_var, False),
            ("Login (account number)", self.login_var, False),
            ("Password", self.pw_var, True),
            ("Server (e.g. Broker-Demo)", self.server_var, False),
        ]
        for r, (label, var, secret) in enumerate(fields, start=1):
            ttk.Label(self.login_frame, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(self.login_frame, textvariable=var, width=34,
                      show="*" if secret else "").grid(row=r, column=1, pady=3)
        ttk.Label(self.login_frame,
                  text="Demo or real. Leave login blank to attach to the terminal's\n"
                       "currently logged-in account.",
                  foreground="#555").grid(row=5, column=0, columnspan=2, pady=(6, 8))
        ttk.Button(self.login_frame, text="Connect",
                   command=self._connect).grid(row=6, column=0, columnspan=2, pady=6)
        if _ENGINE_ERR is not None:
            ttk.Label(self.login_frame,
                      text="(engine import pending until run on the MT5 machine)",
                      foreground="#999").grid(row=7, column=0, columnspan=2)

    def _connect(self):
        try:
            import MetaTrader5 as mt5
        except Exception:
            messagebox.showerror("MetaTrader5 missing",
                                 "The MetaTrader5 package is not installed.\n"
                                 "Run:  pip install MetaTrader5")
            return
        path = self.path_var.get().strip()
        ok = mt5.initialize(path) if path else mt5.initialize()
        if not ok:
            messagebox.showerror("Connect failed",
                                 "mt5.initialize() failed: %s" % (mt5.last_error(),))
            return
        login = self.login_var.get().strip()
        if login:
            try:
                logged = mt5.login(int(login), password=self.pw_var.get(),
                                   server=self.server_var.get().strip())
            except Exception as e:
                messagebox.showerror("Login error", str(e))
                mt5.shutdown()
                return
            if not logged:
                messagebox.showerror("Login failed",
                                     "mt5.login() failed: %s" % (mt5.last_error(),))
                mt5.shutdown()
                return
        if mt5.account_info() is None:
            messagebox.showerror("No account", "Could not read account_info().")
            mt5.shutdown()
            return
        self.mt5 = mt5
        self.login_frame.destroy()
        self._build_dashboard()
        self._start_poller()
        self.root.after(200, self._drain)

    # ---- dashboard ----
    def _build_dashboard(self):
        f = ttk.Frame(self.root, padding=12)
        f.grid(sticky="nsew")
        self.dash = f

        # account panel
        acc = ttk.LabelFrame(f, text="Account", padding=8)
        acc.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.acc_labels = {}
        for r, key in enumerate(["login", "server", "company", "currency", "trade_mode",
                                 "balance", "equity", "margin", "free_margin", "margin_level"]):
            ttk.Label(acc, text=key.replace("_", " ").title()).grid(row=r, column=0, sticky="w")
            v = ttk.Label(acc, text="-", font=("TkDefaultFont", 9, "bold"))
            v.grid(row=r, column=1, sticky="e", padx=(12, 0))
            self.acc_labels[key] = v

        # controls
        ctl = ttk.LabelFrame(f, text="Trade settings", padding=8)
        ctl.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctl, text="Timeframe").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctl, textvariable=self.tf_var, values=["H1", "M15", "M5"],
                     width=8, state="readonly").grid(row=0, column=1, pady=3)
        ttk.Label(ctl, text="Broker").grid(row=1, column=0, sticky="w")
        ttk.Combobox(ctl, textvariable=self.broker_var,
                     values=["headway_cent", "standard"], width=12,
                     state="readonly").grid(row=1, column=1, pady=3)
        ttk.Label(ctl, text="Amount to trade (USD)").grid(row=2, column=0, sticky="w")
        ttk.Entry(ctl, textvariable=self.amount_var, width=10).grid(row=2, column=1, pady=3)
        ttk.Label(ctl, text="(blank = auto-detect balance)",
                  foreground="#777").grid(row=3, column=0, columnspan=2, sticky="w")
        self.start_btn = ttk.Button(ctl, text="Start Trading", command=self._start)
        self.start_btn.grid(row=4, column=0, pady=(8, 0))
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=4, column=1, pady=(8, 0))
        self.status_lbl = ttk.Label(ctl, text="Idle", foreground="#555")
        self.status_lbl.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # signal panel
        sig = ttk.LabelFrame(f, text="Signal", padding=8)
        sig.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        self.sig_head = ttk.Label(sig, text="waiting...", font=("TkDefaultFont", 16, "bold"))
        self.sig_head.grid(row=0, column=0, sticky="w")
        self.sig_why = ttk.Label(sig, text="", wraplength=620, foreground="#333")
        self.sig_why.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.sig_detail = ttk.Label(sig, text="", foreground="#777")
        self.sig_detail.grid(row=2, column=0, sticky="w")

        # trades table
        tr = ttk.LabelFrame(f, text="Trades today", padding=8)
        tr.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        cols = ("pos", "side", "entry_time", "entry_price", "exit_time",
                "exit_price", "volume", "pnl", "status")
        self.tree = ttk.Treeview(tr, columns=cols, show="headings", height=10)
        heads = {"pos": "Position", "side": "Side", "entry_time": "Entry",
                 "entry_price": "Entry px", "exit_time": "Exit", "exit_price": "Exit px",
                 "volume": "Lots", "pnl": "P&L (USD)", "status": "Status"}
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=84, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(tr, orient="vertical", command=self.tree.yview)
        vs.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vs.set)
        self.pnl_lbl = ttk.Label(tr, text="Realised $0.00  |  Floating $0.00  |  Total $0.00",
                                 font=("TkDefaultFont", 11, "bold"))
        self.pnl_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))

        ttk.Label(f, text="PAPER-TRADE on a demo login first. Trades are placed by the "
                          "consolidated engine (1:1 with the backtester).",
                  foreground="#a00").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _ctx(self):
        return self.tf_var.get(), self.broker_var.get()

    def _start_poller(self):
        self.poll_stop = threading.Event()
        _Poller(self._ctx, self.ui_q, self.poll_stop).start()

    # ---- trading control ----
    def _start(self):
        if _run_loop_inner is None:
            messagebox.showerror("Engine unavailable",
                                 "Could not import the trading engine:\n%s" % _ENGINE_ERR)
            return
        info = self.mt5.account_info()
        if info is not None and int(getattr(info, "trade_mode", 0)) == 2:
            if not messagebox.askyesno("REAL account",
                                       "This is a REAL-money account. Start live "
                                       "auto-trading anyway?"):
                return
        tf = self.tf_var.get()
        broker = self.broker_var.get()
        amt = self.amount_var.get().strip()
        try:
            account_size = float(amt) if amt else None
        except ValueError:
            messagebox.showerror("Bad amount", "Amount must be a number (USD) or blank.")
            return
        self.trade_stop = threading.Event()
        smt5 = _StoppableMT5(self.mt5, self.trade_stop)

        def _target():
            try:
                _run_loop_inner(tf, broker, account_size, smt5, None, use_tiered=False)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                self.ui_q.put(("error", "engine stopped: %s" % e))
            self.ui_q.put(("engine_stopped", None))

        self.trade_thread = threading.Thread(target=_target, daemon=True)
        self.trade_thread.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_lbl.config(text="Trading %s (%s)" % (tf, broker), foreground="#0a0")

    def _stop(self):
        if self.trade_stop is not None:
            self.trade_stop.set()
        self.stop_btn.config(state="disabled")
        self.status_lbl.config(text="Stopping...", foreground="#a60")

    # ---- UI queue drain ----
    def _drain(self):
        try:
            while True:
                kind, payload = self.ui_q.get_nowait()
                if kind == "account":
                    self._render_account(payload)
                elif kind == "trades":
                    self._render_trades(*payload)
                elif kind == "signal":
                    self._render_signal(payload)
                elif kind == "engine_stopped":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_lbl.config(text="Idle", foreground="#555")
                elif kind == "error":
                    self.status_lbl.config(text=str(payload), foreground="#a00")
        except queue.Empty:
            pass
        self.root.after(250, self._drain)

    def _render_account(self, tele):
        for key, lbl in self.acc_labels.items():
            val = tele.get(key) if isinstance(tele, dict) else None
            if val is None:
                continue
            if key in ("balance", "equity", "margin", "free_margin"):
                try:
                    val = "%.2f" % float(val)
                except Exception:
                    pass
            lbl.config(text=str(val))

    def _render_signal(self, sig):
        head, why = _describe_signal(sig)
        color = {"BUY": "#0a0", "SELL": "#c00"}.get(head, "#333")
        self.sig_head.config(text=head, foreground=color)
        self.sig_why.config(text=why)
        if sig:
            self.sig_detail.config(text=("regime %s | prob_up %.2f prob_down %.2f | thr %.2f | "
                                         "atr %.2f | close %.2f | exit %s")
                % (sig.get("regime_code", "-"), float(sig.get("prob_up", 0) or 0),
                   float(sig.get("prob_down", 0) or 0), float(sig.get("threshold", 0) or 0),
                   float(sig.get("atr", 0) or 0), float(sig.get("close", 0) or 0),
                   (sig.get("base_params", {}) or {}).get("exit_model", "-")))

    def _render_trades(self, rows, realised, floating):
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            self.tree.insert("", "end", values=(r["pos"], r["side"], r["entry_time"],
                             r["entry_price"], r["exit_time"], r["exit_price"],
                             r["volume"], r["pnl"], r["status"]))
        self.pnl_lbl.config(text="Realised $%.2f  |  Floating $%.2f  |  Total $%.2f"
                            % (realised, floating, realised + floating))

    def on_close(self):
        if self.trade_stop is not None:
            self.trade_stop.set()
        if self.poll_stop is not None:
            self.poll_stop.set()
        try:
            if self.mt5 is not None:
                self.mt5.shutdown()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LiveApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
