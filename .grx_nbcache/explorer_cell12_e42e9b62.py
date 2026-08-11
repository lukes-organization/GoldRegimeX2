# -----------------------------
# Strategy parameter plateau tools
# -----------------------------

def build_coarse_grid():
    # Grid searches ML confidence threshold
    # Lowered to realistic financial ML probability ranges for minority classes
    xgb_threshold_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    return [(float(x),) for x in xgb_threshold_grid]

def build_refined_grid_from_top(
    coarse_df: pd.DataFrame,
    top_k: int = 3,
    step: float = 0.02,
) -> list[tuple[float]]:
    cd = coarse_df.copy()
    cd = cd[np.isfinite(cd["stability_adjusted_sharpe"])].sort_values(
        ["stability_adjusted_sharpe", "mean_sharpe"], ascending=False
    )
    top = cd.head(top_k)

    refined = set()
    for _, r in top.iterrows():
        center = float(r["xgb_threshold"])
        for delta in (-2 * step, -step, 0.0, step, 2 * step):
            cand = round(center + delta, 4)
            # Bounded to match the lowered threshold reality
            if 0.05 <= cand <= 0.45:
                refined.add((float(cand),))

    return sorted(list(refined))

def plot_plateau_heatmaps(results_df: pd.DataFrame, title_prefix: str = "Strategy Surface"):
    if results_df.empty:
        print("No results to plot")
        return

    d = results_df.copy()
    d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["xgb_threshold"])]
    if d.empty:
        print("No finite results to plot")
        return

    d = d.sort_values("xgb_threshold")
    plt.figure(figsize=(8, 4))
    plt.plot(d["xgb_threshold"], d["mean_sharpe"], marker="o", label="mean_sharpe")
    if "stability_adjusted_sharpe" in d.columns:
        plt.plot(d["xgb_threshold"], d["stability_adjusted_sharpe"], marker="s", label="stability_adjusted_sharpe")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"{title_prefix} | xgb_threshold vs score")
    plt.xlabel("xgb_threshold")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_plateau_center(fine_df: pd.DataFrame, min_mean_sharpe: float = -1e9, neighbor_width: float = 0.05) -> dict | None:
    d = fine_df.copy()
    d = d[np.isfinite(d["mean_sharpe"]) & np.isfinite(d["stability_adjusted_sharpe"])]
    d = d[d["mean_sharpe"] >= float(min_mean_sharpe)]
    if d.empty:
        return None

    best = None
    best_score = -1e9

    for _, row in d.iterrows():
        thr = float(row["xgb_threshold"])
        neigh = d[d["xgb_threshold"].between(thr - neighbor_width, thr + neighbor_width)]

        plateau_width = int((neigh["mean_sharpe"] > 0).sum())
        local_stab = float(neigh["stability_adjusted_sharpe"].mean())
        local_mean = float(neigh["mean_sharpe"].mean())
        score = plateau_width * 1.5 + local_stab + 0.15 * local_mean

        if score > best_score:
            best_score = score
            best = {
                "xgb_threshold": thr,
                "mean_sharpe": float(row["mean_sharpe"]),
                "mean_sharpe_raw": float(row.get("mean_sharpe_raw", np.nan)),
                "variance_sharpe": float(row["variance_sharpe"]),
                "stability_adjusted_sharpe": float(row["stability_adjusted_sharpe"]),
                "turnover_penalty": float(row.get("turnover_penalty", 0.0)),
                "mean_trades_per_100": float(row.get("mean_trades_per_100", np.nan)),
                "plateau_width_local": plateau_width,
                "selection_score": float(score),
            }

    return best