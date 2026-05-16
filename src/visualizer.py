import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

REPORT_DIR = Path("reports")
REGIME_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12", 3: "#9b59b6"}
REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "Chop", 3: "Chop_High"}

# Max bars to render in time-series charts.  M5 has 626K bars over 10 years —
# rendering all of them causes weekend gaps to appear as straight horizontal
# lines and makes regime fills look noisy/white.  Downsampling to ~15K bars
# gives ~1 point per hour on M5, equivalent to H1 resolution visually.
_MAX_DISPLAY_BARS = 15_000


def _downsample(df: pd.DataFrame, states: np.ndarray, max_bars: int = _MAX_DISPLAY_BARS):
    """Reduce df and states to at most max_bars rows for display.

    Uses a fixed stride so the time axis stays evenly spaced and weekend gaps
    are no longer wide enough to produce obvious straight-line artefacts.
    Downsampling is purely cosmetic — models are unaffected.
    """
    n = len(df)
    if n <= max_bars:
        return df, states
    step = max(1, n // max_bars)
    return df.iloc[::step], states[::step]


def _tf_dir(tf: str = "H1", broker: str = "headway_cent") -> Path:
    """Return the TF+broker-specific report subdirectory, creating it if needed."""
    d = REPORT_DIR / f"{tf.upper()}_{broker}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _signal_attribution(signals: np.ndarray, hmm_states: np.ndarray,
                         strategy_returns: np.ndarray, chop_state: int = 2):
    """Split per-trade stats into Trend vs Mean-Reversion categories.

    Returns a dict with keys:
        trend_n, trend_win_rate, trend_return
        mr_n, mr_win_rate, mr_return
    """
    prev_sig = np.concatenate([[0], signals[:-1]])
    is_entry = (signals != 0) & (signals != prev_sig)
    trade_id = np.cumsum(is_entry)

    entry_bars = np.where(is_entry)[0]
    trade_type: dict[int, str] = {}
    for bar_idx in entry_bars:
        tid = int(trade_id[bar_idx])
        trade_type[tid] = "trend" if hmm_states[bar_idx] < chop_state else "mr"

    in_trade = signals != 0
    per_trade: dict[int, float] = {}
    if in_trade.any():
        for tid in np.unique(trade_id[in_trade]):
            per_trade[int(tid)] = float(strategy_returns[(trade_id == tid) & in_trade].sum())

    trend_tr = [v for tid, v in per_trade.items() if trade_type.get(tid) == "trend"]
    mr_tr    = [v for tid, v in per_trade.items() if trade_type.get(tid) == "mr"]

    def _stats(trades):
        if not trades:
            return 0, 0.0, 0.0
        wr  = sum(1 for v in trades if v > 0) / len(trades)
        ret = float(np.sum(trades))
        return len(trades), wr, ret

    tn, twr, tr = _stats(trend_tr)
    mn, mwr, mr_r = _stats(mr_tr)
    return {
        "trend_n": tn, "trend_win_rate": twr, "trend_return": tr,
        "mr_n":    mn, "mr_win_rate":    mwr, "mr_return":    mr_r,
    }


def plot_regime_overlay(df, hmm_states, state_names, tf="H1", broker="headway_cent", save_path=None):
    """Price chart with HMM regime shading."""
    save_path = save_path or _tf_dir(tf, broker) / "1_regime_overlay.png"

    df_plot, states_plot = _downsample(df, hmm_states)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.05})

    dates = df_plot.index
    close = df_plot["Close"].values

    for state_id, label in state_names.items():
        mask = states_plot == state_id
        color = REGIME_COLORS.get(state_id, "#95a5a6")
        ax1.fill_between(dates, close.min() * 0.98, close.max() * 1.02,
                         where=mask, alpha=0.25, color=color, label=label)

    ax1.plot(dates, close, color="#2c3e50", linewidth=0.6, alpha=0.9)
    n_total = len(df)
    n_shown = len(df_plot)
    sample_note = (f"  (display: every {n_total // n_shown}th bar — {n_shown:,} of {n_total:,} M5 bars)"
                   if n_shown < n_total else "")
    ax1.set_ylabel("XAUUSD Price", fontsize=11)
    ax1.set_title(f"Gold Regime X — HMM Regime Detection Over Price  [{tf}]{sample_note}",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(dates[0], dates[-1])

    ax2.scatter(dates, states_plot, c=[REGIME_COLORS.get(s, "#95a5a6") for s in states_plot],
                s=2, alpha=0.7)
    ax2.set_ylabel("HMM State", fontsize=11)
    ax2.set_yticks(list(state_names.keys()))
    ax2.set_yticklabels(list(state_names.values()))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved regime overlay chart to %s", save_path)
    return str(save_path)


def plot_equity_curve(df, probabilities, hmm_states, split_idx=None, tf="H1", broker="headway_cent",
                      regime_stats=None, save_path=None):
    """Equity curve with signal-type differentiated entry markers.

    Entry markers are split into four visual categories:
      Blue   triangle-up   = Trend BUY   (HMM Bull state, high XGB buy probability)
      Red    triangle-down = Trend SELL  (HMM Bear state, high XGB sell probability)
      Gold   circle        = MR BUY      (HMM Chop state, XGB buy signal in low-vol cluster)
      Purple circle        = MR SELL     (HMM Chop state, XGB sell signal in high-vol cluster)
    """
    save_path = save_path or _tf_dir(tf, broker) / "2_equity_curve.png"

    from src.backtester import compute_signals, compute_position_sizes, CHOP_STATE

    signals = compute_signals(df, probabilities, hmm_states, tf=tf, broker=broker)
    sizes = compute_position_sizes(signals, df["atr_normalized"].values)

    log_returns  = df["log_return"].values
    next_returns = np.roll(log_returns, -1)
    next_returns[-1] = 0.0
    strategy_returns = sizes * next_returns

    cumulative_strat = np.cumsum(strategy_returns)
    cumulative_bh    = np.cumsum(log_returns)

    running_max = np.maximum.accumulate(cumulative_strat)
    drawdown = running_max - cumulative_strat

    # Detect first bar of each new signal (entries only, not holds)
    prev_sig = np.concatenate([[0], signals[:-1]])
    is_entry = (signals != 0) & (signals != prev_sig)

    trend_buy  = is_entry & (signals == 1)  & (hmm_states < CHOP_STATE)
    trend_sell = is_entry & (signals == -1) & (hmm_states < CHOP_STATE)
    mr_buy     = is_entry & (signals == 1)  & (hmm_states >= CHOP_STATE)
    mr_sell    = is_entry & (signals == -1) & (hmm_states >= CHOP_STATE)

    # Downsample display arrays
    n_total = len(df)
    step    = max(1, n_total // _MAX_DISPLAY_BARS)
    dates        = df.index[::step]
    cum_strat    = cumulative_strat[::step]
    cum_bh       = cumulative_bh[::step]
    dd_plot      = drawdown[::step]
    probs_plot   = probabilities[::step]
    trend_buy_p  = trend_buy[::step]
    trend_sell_p = trend_sell[::step]
    mr_buy_p     = mr_buy[::step]
    mr_sell_p    = mr_sell[::step]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), height_ratios=[3, 1, 1],
                              sharex=True, gridspec_kw={"hspace": 0.08})

    # ── Equity curves ───────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, cum_strat, color="#2980b9", linewidth=1.2, label="Strategy (log return)")
    ax1.plot(dates, cum_bh, color="#95a5a6", linewidth=0.8, alpha=0.7, label="Buy & Hold")

    if trend_buy_p.any():
        ax1.scatter(dates[trend_buy_p], cum_strat[trend_buy_p],
                    marker="^", c="#2980b9", s=35, alpha=0.75, zorder=4,
                    label=f"Trend BUY ({int(trend_buy.sum())})")
    if trend_sell_p.any():
        ax1.scatter(dates[trend_sell_p], cum_strat[trend_sell_p],
                    marker="v", c="#e74c3c", s=35, alpha=0.75, zorder=4,
                    label=f"Trend SELL ({int(trend_sell.sum())})")
    if mr_buy_p.any():
        ax1.scatter(dates[mr_buy_p], cum_strat[mr_buy_p],
                    marker="o", c="#f39c12", s=30, alpha=0.85, zorder=5,
                    label=f"MR BUY ({int(mr_buy.sum())})")
    if mr_sell_p.any():
        ax1.scatter(dates[mr_sell_p], cum_strat[mr_sell_p],
                    marker="o", c="#9b59b6", s=30, alpha=0.85, zorder=5,
                    label=f"MR SELL ({int(mr_sell.sum())})")

    # Train/test split line
    all_dates = df.index
    if split_idx is not None and 0 < split_idx < len(all_dates):
        split_date = all_dates[split_idx]
        for ax in axes:
            ax.axvline(x=split_date, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
        ax1.axvspan(split_date, dates[-1], alpha=0.04, color="#e74c3c")
        ax1.text(split_date, ax1.get_ylim()[1] * 0.95, "  OOS ->", fontsize=10,
                 color="#e74c3c", fontweight="bold", va="top")
        ax1.text(split_date, ax1.get_ylim()[1] * 0.95, "<- Train  ", fontsize=10,
                 color="#2980b9", fontweight="bold", va="top", ha="right")

    ax1.set_ylabel("Cumulative Log Return", fontsize=11)
    ax1.set_title("Gold Regime X — Strategy Equity Curve vs Buy & Hold", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)

    # ── Drawdown ────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(dates, 0, -dd_plot * 100, color="#e74c3c", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    max_dd_idx = int(np.argmax(dd_plot))
    ax2.annotate(f"Max DD: {dd_plot[max_dd_idx]*100:.1f}%",
                 xy=(dates[max_dd_idx], -dd_plot[max_dd_idx] * 100),
                 fontsize=9, color="#c0392b", fontweight="bold")

    # ── XGB probability ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.scatter(dates, probs_plot, c=probs_plot, cmap="RdYlGn", s=2, alpha=0.4, vmin=0.3, vmax=0.7)
    ax3.axhline(y=0.5, color="#7f8c8d", linewidth=1, linestyle="--", alpha=0.7, label="Mid (0.50)")
    ax3.set_ylabel("XGB Probability", fontsize=11)
    ax3.set_ylim(0.25, 0.75)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved equity curve chart to %s", save_path)
    return str(save_path)


def plot_feature_analysis(X, hmm_states, metrics, tf="H1", broker="headway_cent", save_path=None):
    """Feature importance, RSI regime distributions, regime scatter, and pie."""
    save_path = save_path or _tf_dir(tf, broker) / "3_feature_analysis.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Top-left: XGBoost feature importance ────────────────────────────────
    ax = axes[0, 0]
    importance = metrics.get("feature_importance", {})
    names  = list(importance.keys())
    values = [float(v) for v in importance.values()]
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#d35400"]
    bars = ax.barh(names, values, color=colors[:len(names)])
    ax.set_xlabel("Importance (Gain)", fontsize=10)
    ax.set_title("XGBoost Feature Importance", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # ── Top-right: RSI slope distribution by regime ──────────────────────────
    ax = axes[0, 1]
    unique_states = sorted(np.unique(hmm_states))
    for s in unique_states:
        mask = hmm_states == s
        data = X.loc[mask, "rsi_slope"].dropna() if "rsi_slope" in X.columns else pd.Series(dtype=float)
        if len(data):
            ax.hist(data, bins=80, alpha=0.5, label=REGIME_LABELS.get(s, str(s)),
                    color=REGIME_COLORS.get(s, "#95a5a6"), density=True)
    ax.set_xlabel("RSI Slope", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("RSI Slope Distribution by Regime", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: 2D regime scatter (RSI Slope vs ATR) ───────────────────
    # This directly visualises how the 3-feature HMM separates Trend from Chop
    # using momentum (rsi_slope) as the third axis.
    ax = axes[1, 0]
    n_feat  = len(X)
    step_s  = max(1, n_feat // 6000)   # cap scatter pts for readability
    X_sub   = X.iloc[::step_s]
    st_sub  = hmm_states[::step_s] if isinstance(hmm_states, np.ndarray) else hmm_states.values[::step_s]
    if "rsi_slope" in X.columns and "atr_normalized" in X.columns:
        for s in sorted(np.unique(st_sub)):
            mask = st_sub == s
            ax.scatter(
                X_sub.loc[X_sub.index[np.where(mask)[0]], "rsi_slope"].clip(-8, 8),
                X_sub.loc[X_sub.index[np.where(mask)[0]], "atr_normalized"],
                c=REGIME_COLORS.get(int(s), "#95a5a6"), alpha=0.25, s=5,
                label=REGIME_LABELS.get(int(s), str(s)),
            )
        ax.set_xlabel("RSI Slope   (momentum — HMM feature 3)", fontsize=10)
        ax.set_ylabel("ATR Normalized   (volatility — HMM feature 2)", fontsize=10)
        ax.set_title("Regime Clusters: RSI Slope vs ATR", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, markerscale=3)
        ax.set_xlim(-8, 8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "rsi_slope / atr_normalized\nnot in feature matrix",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#7f8c8d")
        ax.axis("off")

    # ── Bottom-right: Regime distribution pie ───────────────────────────────
    ax = axes[1, 1]
    state_counts = pd.Series(hmm_states).value_counts().sort_index()
    labels_pie   = [REGIME_LABELS.get(s, str(s)) for s in state_counts.index]
    colors_pie   = [REGIME_COLORS.get(s, "#95a5a6") for s in state_counts.index]
    wedges, texts, autotexts = ax.pie(
        state_counts.values, labels=labels_pie, colors=colors_pie,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Regime Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature analysis chart to %s", save_path)
    return str(save_path)


def plot_transition_matrix(model_hmm, state_names, tf="H1", broker="headway_cent", save_path=None):
    """HMM transition matrix heatmap."""
    save_path = save_path or _tf_dir(tf, broker) / "4_transition_matrix.png"

    n     = model_hmm.n_components
    trans = model_hmm.transmat_

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(trans, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    labels = [state_names.get(i, str(i)) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)
    ax.set_title("HMM Transition Matrix", fontsize=14, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if trans[i, j] > 0.5 else "black"
            ax.text(j, i, f"{trans[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Transition Probability")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved transition matrix chart to %s", save_path)
    return str(save_path)


def plot_summary_dashboard(result, params, tf="H1", broker="headway_cent",
                           account_size: float = 15.0, save_path=None):
    """Single-panel summary card with key metrics, attribution table, and params."""
    save_path = save_path or _tf_dir(tf, broker) / "5_summary_dashboard.png"

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    # ── Title block ──────────────────────────────────────────────────────────
    ax.text(0.50, 0.97, "GOLD REGIME X", fontsize=20,
            fontweight="bold", transform=ax.transAxes, va="top", ha="center", color="#2c3e50")
    ax.text(0.50, 0.91, f"Hybrid HMM + XGBoost  |  XAUUSD {tf.upper()}  |  ${account_size:.0f} account",
            fontsize=12, transform=ax.transAxes, va="top", ha="center", color="#7f8c8d")
    ax.plot([0.05, 0.95], [0.87, 0.87], color="#bdc3c7", linewidth=1, transform=ax.transAxes)

    # ── Left column: Full-period performance ─────────────────────────────────
    fdd      = result.get("floating_max_drawdown", result.get("max_drawdown", 0.0))
    fdd_usd  = fdd * account_size
    sharpe   = result["sharpe_ratio"]
    wr       = result["win_rate"]
    ret      = result["total_return"]
    rf       = result.get("recovery_factor", 0.0)
    pf       = result.get("profit_factor", 1.0)

    _period_label = "Performance (Full Period IS)" if "cpcv_score" in result else "Performance (Full Period)"
    ax.text(0.05, 0.83, _period_label, fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top", color="#2c3e50")
    metrics_data = [
        ("Sharpe Ratio",    f"{sharpe:.3f}",  "#2ecc71" if sharpe >= 1.0 else "#e67e22"),
        ("Recovery Factor", f"{rf:.2f}",      "#2ecc71" if rf >= 1.5 else "#e67e22"),
        ("Profit Factor",   f"{pf:.2f}",      "#2ecc71" if pf >= 1.2 else "#e67e22"),
        ("Win Rate",        f"{wr*100:.1f}%", "#3498db"),
        ("Total Return",    f"{ret*100:.1f}%","#3498db"),
        ("Trade Count",     f"{result['n_trades']}", "#3498db"),
        ("Max Float DD",    f"{fdd*100:.1f}%  (~${fdd_usd:.2f})", "#e74c3c" if fdd > 0.10 else "#e67e22"),
    ]
    y = 0.75
    for label, value, color in metrics_data:
        ax.text(0.07, y, label, fontsize=11, transform=ax.transAxes, va="top",
                fontfamily="monospace", color="#2c3e50")
        ax.text(0.33, y, value, fontsize=11, transform=ax.transAxes, va="top",
                fontfamily="monospace", fontweight="bold", color=color)
        y -= 0.075

    # ── OOS metrics ──────────────────────────────────────────────────────────
    if "oos_sharpe_ratio" in result:
        y -= 0.01
        _is_cpcv = "cpcv_score" in result
        _oos_label = (
            f"CPCV OOS  ({result.get('cpcv_n_valid_paths', '?')}/{6} valid paths)"
            if _is_cpcv else "Out-of-Sample"
        )
        ax.text(0.05, y, _oos_label, fontsize=12, fontweight="bold",
                transform=ax.transAxes, va="top", color="#c0392b")
        y -= 0.06
        oos_fdd = result.get("oos_floating_max_drawdown", result.get("oos_max_drawdown", 0.0))
        oos_fdd_usd = oos_fdd * account_size
        oos_data = [
            ("Sharpe (median)",  f"{result['oos_sharpe_ratio']:.3f}"),
            ("Std Sharpe",       f"{result.get('cpcv_std_sharpe', 0.0):.3f}" if _is_cpcv else "N/A"),
            ("Max DD (median)",  f"{oos_fdd*100:.1f}%"),
            ("Max Monetary Risk",f"${oos_fdd_usd:.2f} USD"),
            ("Win Rate (median)",f"{result['oos_win_rate']*100:.1f}%"),
            ("Trades (median)",  f"{result['oos_n_trades']}"),
        ]
        for label, value in oos_data:
            ax.text(0.07, y, label, fontsize=10, transform=ax.transAxes, va="top",
                    fontfamily="monospace", color="#7f8c8d")
            ax.text(0.30, y, value, fontsize=10, transform=ax.transAxes, va="top",
                    fontfamily="monospace", fontweight="bold", color="#c0392b")
            y -= 0.060

    # ── Middle column: Signal-type attribution ───────────────────────────────
    ax.text(0.50, 0.83, "Profit Attribution", fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top", color="#2c3e50")
    ax.plot([0.48, 0.73], [0.80, 0.80], color="#bdc3c7", linewidth=0.7, transform=ax.transAxes)

    trend_n  = result.get("trend_n", 0)
    trend_wr = result.get("trend_win_rate", 0.0)
    trend_r  = result.get("trend_return", 0.0)
    mr_n     = result.get("mr_n", 0)
    mr_wr    = result.get("mr_win_rate", 0.0)
    mr_r     = result.get("mr_return", 0.0)

    def _pnl_str(log_ret):
        """Convert log-return fraction to approximate USD P&L string."""
        approx = (float(np.exp(log_ret)) - 1.0) * account_size
        sign = "+" if approx >= 0 else ""
        return f"{sign}${approx:.2f}"

    attr_rows = [
        ("",          "Signal Type", "WR",           "P&L (approx)"),
        ("Trend",     f"{trend_n} trades", f"{trend_wr*100:.0f}%", _pnl_str(trend_r)),
        ("MR (Fade)", f"{mr_n} trades",    f"{mr_wr*100:.0f}%",    _pnl_str(mr_r)),
    ]
    col_xs = [0.50, 0.60, 0.67, 0.73]
    y_attr = 0.77
    for row_idx, row in enumerate(attr_rows):
        for col_idx, cell in enumerate(row):
            weight = "bold" if row_idx == 0 else "normal"
            fsize  = 9 if row_idx == 0 else 10
            color  = "#7f8c8d" if row_idx == 0 else "#2c3e50"
            if row_idx > 0 and col_idx == 3:
                val = float(np.exp(trend_r if row_idx == 1 else mr_r)) - 1.0
                color = "#2ecc71" if val >= 0 else "#e74c3c"
                weight = "bold"
            ax.text(col_xs[col_idx], y_attr, cell, fontsize=fsize,
                    transform=ax.transAxes, va="top", fontfamily="monospace",
                    fontweight=weight, color=color)
        y_attr -= 0.07

    # Attribution bar chart
    if trend_n + mr_n > 0:
        y_bar = y_attr - 0.01
        bar_w = 0.20
        bar_h = 0.065
        trend_frac = trend_n / (trend_n + mr_n) if (trend_n + mr_n) > 0 else 0.5
        ax.add_patch(plt.Rectangle((0.50, y_bar), bar_w * trend_frac, bar_h,
                                    transform=ax.transAxes, color="#2980b9", alpha=0.7, zorder=2))
        ax.add_patch(plt.Rectangle((0.50 + bar_w * trend_frac, y_bar),
                                    bar_w * (1 - trend_frac), bar_h,
                                    transform=ax.transAxes, color="#f39c12", alpha=0.7, zorder=2))
        ax.text(0.50, y_bar - 0.025, "Trade mix: Trend (blue) vs MR (gold)",
                fontsize=8, transform=ax.transAxes, va="top", color="#7f8c8d")

    # ── Right column: Optimized params ───────────────────────────────────────
    ax.text(0.78, 0.83, "Optimized Params", fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top", color="#2c3e50")
    ax.text(0.78, 0.79, "Target Ranges", fontsize=10,
            transform=ax.transAxes, va="top", color="#7f8c8d")
    targets = ["Sharpe: >= 1.0", "Max DD: <= 15%", "Win Rate: 50-55%", "PF: >= 1.2"]
    y_right = 0.73
    for t in targets:
        ax.text(0.80, y_right, t, fontsize=9, transform=ax.transAxes, va="top",
                fontfamily="monospace", color="#7f8c8d")
        y_right -= 0.065

    if params:
        y_right -= 0.02
        ax.text(0.78, y_right, "Parameters", fontsize=10, fontweight="bold",
                transform=ax.transAxes, va="top", color="#2c3e50")
        y_right -= 0.06
        for k, v in params.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            ax.text(0.80, y_right, f"{k}: {val_str}", fontsize=8, transform=ax.transAxes,
                    va="top", fontfamily="monospace", color="#34495e")
            y_right -= 0.05

    # ── Border ───────────────────────────────────────────────────────────────
    fig.patch.set_facecolor("#fafafa")
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#bdc3c7",
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    logger.info("Saved summary dashboard to %s", save_path)
    return str(save_path)


def plot_mt5_equity_curve(
    result: dict,
    account_size: float = 15.0,
    split_idx: int = None,
    tf: str = "H1",
    broker: str = "headway_cent",
) -> Path:
    """MT5 Strategy Tester-style Balance/Equity vs time chart.

    Top panel  — Balance (blue step-line) and Equity (teal continuous line)
                 with an optional IS/OOS divider.
    Bottom panel — Deposit Load: shaded fill showing when a position is open
                   (100%) vs flat (0%), mirroring the MT5 margin-usage strip.

    Balance is sourced from ``result["balance_values"]`` (staircase that only
    steps at trade close) and Equity from ``result["equity_values"]``
    (continuous cumulative P&L).  Both arrays are injected by
    ``vectorized_backtest``; the function is a safe no-op when absent.
    """
    timestamps     = result.get("equity_timestamps")
    equity_values  = result.get("equity_values")
    balance_values = result.get("balance_values")
    deposit_load   = result.get("deposit_load")
    if timestamps is None or equity_values is None:
        logger.warning("Equity series not in result — skipping MT5 chart.")
        return None

    dates = pd.to_datetime(timestamps)

    # ── MT5 dark theme ────────────────────────────────────────────────────────
    BG      = "#131722"
    GRID    = "#1e2535"
    BAL_CLR = "#2196f3"   # blue   — Balance
    EQ_CLR  = "#26a69a"   # teal   — Equity
    DL_CLR  = "#546e7a"   # slate  — Deposit Load fill
    TXT     = "#d1d4dc"
    SPLIT   = "#f59f00"   # amber  — IS/OOS divider

    fig = plt.figure(figsize=(16, 6), facecolor=BG)
    gs  = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.04)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors=TXT, labelsize=9)

    # Balance + Equity
    ax1.plot(dates, balance_values, color=BAL_CLR, linewidth=1.3,
             label="Balance", drawstyle="steps-post")
    ax1.plot(dates, equity_values,  color=EQ_CLR,  linewidth=0.9,
             label="Equity",  alpha=0.85)

    # IS/OOS split line
    if split_idx and 0 < split_idx < len(dates):
        split_date = dates[split_idx]
        ax1.axvline(split_date, color=SPLIT, linewidth=1.0,
                    linestyle="--", alpha=0.8, label="IS / OOS")
        ax2.axvline(split_date, color=SPLIT, linewidth=1.0,
                    linestyle="--", alpha=0.8)
        # Label just above the x-axis
        ax2.text(split_date, ax2.get_ylim()[1] * 0.9 if ax2.get_ylim()[1] > 0 else 0.9,
                 "OOS", color=SPLIT, fontsize=7, ha="left", va="top")

    # Deposit Load
    if deposit_load is not None:
        pct = deposit_load * 100.0
        ax2.fill_between(dates, pct, 0, color=DL_CLR, alpha=0.75, step="post")
        ax2.set_ylim(0, 120)
        ax2.set_yticks([0, 100])
        ax2.set_yticklabels(["0.0%", "100%"], color=TXT, fontsize=8)

    ax1.set_ylabel("USD", color=TXT, fontsize=9)
    ax2.set_ylabel("Deposit\nLoad", color=TXT, fontsize=8)

    # Legend
    leg = ax1.legend(facecolor="#1e2535", edgecolor=GRID,
                     labelcolor=TXT, fontsize=9, loc="upper left")

    # Legend label in top-left corner like MT5
    ax1.text(0.01, 0.97, "Balance / Equity",
             transform=ax1.transAxes, color=EQ_CLR,
             fontsize=9, va="top")

    title = (
        f"Balance / Equity — {tf.upper()} [{broker}]"
        f"   Initial: ${account_size:.0f}"
        f"   Final balance: ${float(balance_values[-1]):.2f}"
        f"   Final equity: ${float(equity_values[-1]):.2f}"
    )
    ax1.set_title(title, color=TXT, fontsize=9, pad=6)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    out_path = _tf_dir(tf, broker) / "6_balance_equity.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Balance/Equity chart saved: %s", out_path)
    return out_path


def generate_full_report(df, hmm_states, state_names, model_hmm,
                         X, probabilities, metrics, result, params=None,
                         split_idx=None, tf="H1", broker="headway_cent",
                         account_size: float = 15.0):
    """Generate all 6 charts into reports/<TF>_<broker>/ and return list of file paths.

    account_size is used to convert drawdown fractions and signal-attribution
    log-returns into approximate USD figures on the summary dashboard.
    """
    from src.backtester import compute_signals, compute_position_sizes

    # Compute attribution metrics and inject into an enriched copy of result
    # so plot_summary_dashboard can render the Trend vs MR breakdown.
    try:
        _sigs = compute_signals(df, probabilities, hmm_states, tf=tf, broker=broker)
        _sizes = compute_position_sizes(_sigs, df["atr_normalized"].values)
        _lr    = df["log_return"].values
        _nr    = np.roll(_lr, -1)
        _nr[-1] = 0.0
        _sr    = _sizes * _nr
        _attr  = _signal_attribution(_sigs, hmm_states, _sr)
    except Exception as _exc:
        logger.warning("Signal attribution failed (%s) — dashboard will show zeros.", _exc)
        _attr = {"trend_n": 0, "trend_win_rate": 0.0, "trend_return": 0.0,
                 "mr_n": 0, "mr_win_rate": 0.0, "mr_return": 0.0}

    result_enriched = dict(result)
    result_enriched.update(_attr)

    paths = []
    paths.append(plot_regime_overlay(df, hmm_states, state_names, tf=tf, broker=broker))
    paths.append(plot_equity_curve(df, probabilities, hmm_states, split_idx=split_idx,
                                   tf=tf, broker=broker))
    paths.append(plot_feature_analysis(X, hmm_states, metrics, tf=tf, broker=broker))
    paths.append(plot_transition_matrix(model_hmm, state_names, tf=tf, broker=broker))
    paths.append(plot_summary_dashboard(result_enriched, params or {},
                                        tf=tf, broker=broker, account_size=account_size))
    paths.append(plot_mt5_equity_curve(result, account_size=account_size,
                                       split_idx=split_idx, tf=tf, broker=broker))
    logger.info("Full report [%s/%s]: %d charts in %s", tf, broker, len(paths),
                REPORT_DIR / f"{tf.upper()}_{broker}")
    return paths
