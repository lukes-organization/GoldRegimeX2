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


def plot_regime_overlay(df, hmm_states, state_names, tf="H1", broker="headway_cent", save_path=None):
    """Price chart with HMM regime shading."""
    save_path = save_path or _tf_dir(tf, broker) / "1_regime_overlay.png"

    # Downsample for display — avoids straight-line weekend-gap artefacts and
    # noisy regime fills that appear white at high bar density (e.g. M5 10yr).
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


def plot_equity_curve(df, probabilities, hmm_states, split_idx=None, tf="H1", broker="headway_cent", save_path=None):
    """Equity curve, drawdown, and signal markers with train/test split."""
    save_path = save_path or _tf_dir(tf, broker) / "2_equity_curve.png"

    from src.backtester import compute_signals, compute_position_sizes

    signals = compute_signals(probabilities, hmm_states)
    sizes = compute_position_sizes(signals, df["atr_normalized"].values)

    log_returns = df["log_return"].values
    next_returns = np.roll(log_returns, -1)
    next_returns[-1] = 0.0
    strategy_returns = sizes * next_returns

    cumulative_strat = np.cumsum(strategy_returns)
    cumulative_bh = np.cumsum(log_returns)

    running_max = np.maximum.accumulate(cumulative_strat)
    drawdown = running_max - cumulative_strat

    # Downsample display arrays — same reason as regime overlay
    n_total = len(df)
    step = max(1, n_total // _MAX_DISPLAY_BARS)
    dates      = df.index[::step]
    cum_strat  = cumulative_strat[::step]
    cum_bh     = cumulative_bh[::step]
    dd_plot    = drawdown[::step]
    probs_plot = probabilities[::step]
    sig_plot   = signals[::step]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), height_ratios=[3, 1, 1],
                              sharex=True, gridspec_kw={"hspace": 0.08})

    # Equity curves
    ax1 = axes[0]
    ax1.plot(dates, cum_strat, color="#2980b9", linewidth=1.2, label="Strategy (log return)")
    ax1.plot(dates, cum_bh, color="#95a5a6", linewidth=0.8, alpha=0.7, label="Buy & Hold")
    entry_mask = sig_plot == 1
    n_entries = int(np.sum(signals == 1))   # use full signals for true count
    ax1.scatter(dates[entry_mask], cum_strat[entry_mask],
                c="#2ecc71", s=5, alpha=0.6, label=f"Long entries ({n_entries})", zorder=3)

    # Train/test split line
    all_dates = df.index
    if split_idx is not None and 0 < split_idx < len(all_dates):
        split_date = all_dates[split_idx]
        for ax in axes:
            ax.axvline(x=split_date, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
        ax1.axvspan(split_date, dates[-1], alpha=0.04, color="#e74c3c")
        ax1.text(split_date, ax1.get_ylim()[1] * 0.95, "  OOS \u2192", fontsize=10,
                 color="#e74c3c", fontweight="bold", va="top")
        ax1.text(split_date, ax1.get_ylim()[1] * 0.95, "\u2190 Train  ", fontsize=10,
                 color="#2980b9", fontweight="bold", va="top", ha="right")

    ax1.set_ylabel("Cumulative Log Return", fontsize=11)
    ax1.set_title("Gold Regime X \u2014 Strategy Equity Curve vs Buy & Hold", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(dates, 0, -dd_plot * 100, color="#e74c3c", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    max_dd_idx = int(np.argmax(dd_plot))
    ax2.annotate(f"Max DD: {dd_plot[max_dd_idx]*100:.1f}%",
                 xy=(dates[max_dd_idx], -dd_plot[max_dd_idx] * 100),
                 fontsize=9, color="#c0392b", fontweight="bold")

    # XGB probability
    ax3 = axes[2]
    ax3.scatter(dates, probs_plot, c=probs_plot, cmap="RdYlGn", s=2, alpha=0.4, vmin=0.3, vmax=0.7)
    ax3.axhline(y=0.65, color="#2ecc71", linewidth=1, linestyle="--", alpha=0.8, label="Long threshold (0.65)")
    ax3.axhline(y=0.35, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.8, label="Short threshold (0.35)")
    ax3.axhline(y=0.5, color="#7f8c8d", linewidth=0.5, linestyle=":", alpha=0.5)
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
    """Feature importance, distributions per regime, and correlation."""
    save_path = save_path or _tf_dir(tf, broker) / "3_feature_analysis.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    importance = metrics.get("feature_importance", {})
    names = list(importance.keys())
    values = [float(v) for v in importance.values()]
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
    bars = ax.barh(names, values, color=colors[:len(names)])
    ax.set_xlabel("Importance (Gain)", fontsize=10)
    ax.set_title("XGBoost Feature Importance", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[0, 1]
    unique_states = sorted(np.unique(hmm_states))
    for s in unique_states:
        mask = hmm_states == s
        data = X.loc[mask, "rsi_slope"].dropna()
        ax.hist(data, bins=80, alpha=0.5, label=REGIME_LABELS.get(s, str(s)),
                color=REGIME_COLORS.get(s, "#95a5a6"), density=True)
    ax.set_xlabel("RSI Slope", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("RSI Slope Distribution by Regime", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for s in unique_states:
        mask = hmm_states == s
        data = X.loc[mask, "atr_normalized"].dropna()
        ax.hist(data, bins=80, alpha=0.5, label=REGIME_LABELS.get(s, str(s)),
                color=REGIME_COLORS.get(s, "#95a5a6"), density=True)
    ax.set_xlabel("ATR Normalized", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("ATR (Normalized) Distribution by Regime", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    state_counts = pd.Series(hmm_states).value_counts().sort_index()
    labels_pie = [REGIME_LABELS.get(s, str(s)) for s in state_counts.index]
    colors_pie = [REGIME_COLORS.get(s, "#95a5a6") for s in state_counts.index]
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

    n = model_hmm.n_components
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


def plot_summary_dashboard(result, params, tf="H1", broker="headway_cent", save_path=None):
    """Single-panel summary card with key metrics and params."""
    save_path = save_path or _tf_dir(tf, broker) / "5_summary_dashboard.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Title block
    ax.text(0.50, 0.96, "GOLD REGIME X", fontsize=20,
            fontweight="bold", transform=ax.transAxes, va="top", ha="center", color="#2c3e50")
    ax.text(0.50, 0.90, f"Hybrid HMM + XGBoost | XAUUSD {tf.upper()}", fontsize=12,
            transform=ax.transAxes, va="top", ha="center", color="#7f8c8d")

    # Divider line
    ax.plot([0.05, 0.95], [0.86, 0.86], color="#bdc3c7", linewidth=1, transform=ax.transAxes)

    # Left column: Full-period metrics
    metrics_data = [
        ["Sharpe Ratio", f"{result['sharpe_ratio']:.3f}"],
        ["Max Drawdown", f"{result['max_drawdown']*100:.1f}%"],
        ["Win Rate", f"{result['win_rate']*100:.1f}%"],
        ["Total Return", f"{result['total_return']*100:.1f}%"],
        ["Trade Count", f"{result['n_trades']}"],
    ]
    sharpe_color = "#2ecc71" if 1.0 <= result["sharpe_ratio"] <= 1.5 else "#e67e22"
    dd_color = "#2ecc71" if 0.08 <= result["max_drawdown"] <= 0.15 else "#e67e22"
    wr_color = "#2ecc71" if 0.50 <= result["win_rate"] <= 0.55 else "#e67e22"
    metric_colors = [sharpe_color, dd_color, wr_color, "#3498db", "#3498db"]

    ax.text(0.05, 0.82, "Performance (Full Period)", fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top", color="#2c3e50")
    y = 0.74
    for i, (label, value) in enumerate(metrics_data):
        ax.text(0.08, y, label, fontsize=12, transform=ax.transAxes, va="top",
                fontfamily="monospace", color="#2c3e50")
        ax.text(0.35, y, value, fontsize=12, transform=ax.transAxes, va="top",
                fontfamily="monospace", fontweight="bold", color=metric_colors[i])
        y -= 0.085

    # OOS metrics if available
    if "oos_sharpe_ratio" in result:
        y -= 0.02
        ax.text(0.05, y, "Out-of-Sample (Test 20%)", fontsize=13, fontweight="bold",
                transform=ax.transAxes, va="top", color="#c0392b")
        y -= 0.08
        oos_data = [
            ["Sharpe", f"{result['oos_sharpe_ratio']:.3f}"],
            ["Max DD", f"{result['oos_max_drawdown']*100:.1f}%"],
            ["Win Rate", f"{result['oos_win_rate']*100:.1f}%"],
            ["Trades", f"{result['oos_n_trades']}"],
        ]
        for label, value in oos_data:
            ax.text(0.08, y, label, fontsize=11, transform=ax.transAxes, va="top",
                    fontfamily="monospace", color="#7f8c8d")
            ax.text(0.25, y, value, fontsize=11, transform=ax.transAxes, va="top",
                    fontfamily="monospace", fontweight="bold", color="#c0392b")
            y -= 0.07

    # Right column: Target ranges + Params
    ax.text(0.55, 0.82, "Target Ranges", fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top", color="#2c3e50")
    targets = [
        "Sharpe:   1.0 - 1.5",
        "Max DD:   8% - 15%",
        "Win Rate: 50% - 55%",
    ]
    y_right = 0.74
    for t in targets:
        ax.text(0.58, y_right, t, fontsize=11, transform=ax.transAxes, va="top",
                fontfamily="monospace", color="#7f8c8d")
        y_right -= 0.07

    if params:
        y_right -= 0.04
        ax.text(0.55, y_right, "Optimized Params", fontsize=13, fontweight="bold",
                transform=ax.transAxes, va="top", color="#2c3e50")
        y_right -= 0.08
        for k, v in params.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            ax.text(0.58, y_right, f"{k}: {val_str}", fontsize=9, transform=ax.transAxes,
                    va="top", fontfamily="monospace", color="#34495e")
            y_right -= 0.055

    # Border
    fig.patch.set_facecolor("#fafafa")
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#bdc3c7",
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    logger.info("Saved summary dashboard to %s", save_path)
    return str(save_path)


def generate_full_report(df, hmm_states, state_names, model_hmm,
                         X, probabilities, metrics, result, params=None,
                         split_idx=None, tf="H1", broker="headway_cent"):
    """Generate all 5 charts into reports/<TF>_<broker>/ and return list of file paths."""
    paths = []
    paths.append(plot_regime_overlay(df, hmm_states, state_names, tf=tf, broker=broker))
    paths.append(plot_equity_curve(df, probabilities, hmm_states, split_idx=split_idx, tf=tf, broker=broker))
    paths.append(plot_feature_analysis(X, hmm_states, metrics, tf=tf, broker=broker))
    paths.append(plot_transition_matrix(model_hmm, state_names, tf=tf, broker=broker))
    paths.append(plot_summary_dashboard(result, params or {}, tf=tf, broker=broker))
    logger.info("Full report [%s/%s]: %d charts in %s", tf, broker, len(paths),
                REPORT_DIR / f"{tf.upper()}_{broker}")
    return paths
