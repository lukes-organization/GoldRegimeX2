# -----------------------------
# Optional live risk circuit breaker prototype
# -----------------------------

from scipy.stats import entropy

class ProductionRiskCircuitBreaker:
    def __init__(self, historical_mean_sharpe, historical_min_sharpe, max_allowed_slippage_pips=2.5):
        self.hist_mean_sharpe = float(historical_mean_sharpe)
        self.hist_min_sharpe = float(historical_min_sharpe)
        self.max_slippage = float(max_allowed_slippage_pips)

        self.trade_returns = []
        self.trade_slippages = []
        self.hmm_state_probabilities = []

    def log_executed_trade(self, return_pct, slippage_pips, hmm_proba_vector):
        self.trade_returns.append(float(return_pct))
        self.trade_slippages.append(float(slippage_pips))
        self.hmm_state_probabilities.append(np.array(hmm_proba_vector, dtype=float))

    def calculate_rolling_sharpe(self, window=30):
        if len(self.trade_returns) < window:
            return self.hist_mean_sharpe
        r = np.array(self.trade_returns[-window:], dtype=float)
        return float((r.mean() / (r.std() + 1e-6)) * np.sqrt(252.0))

    def check_regime_drift(self, window=12):
        if len(self.hmm_state_probabilities) < window:
            return 0.0
        recent = self.hmm_state_probabilities[-window:]
        return float(np.mean([entropy(np.clip(p, 1e-12, 1.0)) for p in recent]))

    def evaluate_system_health(self):
        if len(self.trade_returns) < 15:
            return "GREEN", "Initialization phase"

        live_sharpe = self.calculate_rolling_sharpe(window=30)
        avg_slippage = float(np.mean(self.trade_slippages[-15:]))
        avg_entropy = self.check_regime_drift(window=12)

        if avg_slippage > self.max_slippage:
            return "RED", f"HALT: slippage {avg_slippage:.2f} exceeds limit"

        if live_sharpe < self.hist_min_sharpe:
            return "RED", f"HALT: live sharpe {live_sharpe:.2f} below minimum"

        if live_sharpe < (self.hist_mean_sharpe * 0.5) or avg_entropy > 1.2:
            return "YELLOW", f"WARN: decay or high entropy {avg_entropy:.2f}"

        return "GREEN", f"Healthy. live sharpe={live_sharpe:.2f}"