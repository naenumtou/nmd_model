
import numpy as np
import pandas as pd
import warnings

from scipy import stats

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Confidence Interval
def stable_ci_method(
    balance: pd.Series,
    confidence: float = 0.95
) -> dict:

    """
    Decompose deposit balance into stable and non-stable using Confidence Interval method.

    Description:
        Step 1: Regress balance against a linear time trend to obtain fitted values.
        Step 2: Construct a lower confidence interval of the mean around the
                regression line. This represents the statistically supported
                minimum expected trend level (not a single-observation prediction).
        Step 3: Core (stable) balance = lower CI at each point in time.
                Non-core = actual balance - core balance.
                Stable % = mean(core) / mean(actual).

        Note: Uses confidence interval of the mean (not prediction interval).
        Prediction intervals are wider because they account for individual
        observation noise, which would understate the stable portion.

    Args:
        balance (pd.Series)     : Monthly deposit balance series, indexed by date.
        confidence (float)      : Confidence level for the interval (default: 0.95).

    Returns:
        dict: {
            "fitted"      : Series — OLS fitted trend values,
            "core"        : Series — lower CI of mean (stable floor),
            "non_core"    : Series — volatile portion (actual - core),
            "stable_pct"  : float  — stable % of total balance,
            "alpha"       : float  — OLS intercept,
            "beta"        : float  — OLS time coefficient
        }
    """

    n = len(balance)
    t = np.arange(n)
    y = balance.values

    X = np.column_stack([np.ones(n), t])
    beta = np.linalg.lstsq(X, y, rcond = None)[0]
    fitted = X @ beta
    residuals = y - fitted

    s = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = stats.t.ppf((1 + confidence) / 2, df = n - 2)

    t_mean = t.mean()
    sxx = np.sum((t - t_mean) ** 2)

    # Confidence interval of the mean (narrower than prediction interval)
    se_mean = s * np.sqrt(1 / n + (t - t_mean) ** 2 / sxx)
    lower_ci = fitted - t_crit * se_mean
    lower_ci = np.maximum(lower_ci, 0)

    core = pd.Series(lower_ci, index = balance.index, name = "core")
    non_core = (balance - core).clip(lower = 0)
    non_core.name = "non_core"

    return {
        "fitted": pd.Series(fitted, index=balance.index, name="fitted"),
        "core": core,
        "non_core": non_core,
        "stable_pct": float(core.mean() / balance.mean()),
        "alpha": float(beta[0]),
        "beta": float(beta[1])
    }

# HP Filter Model
def stable_hp_filter(
    balance: pd.Series,
    lam: float = 1600.0,
    confidence: float = 0.90
) -> dict:

    """
    Decompose deposit balance into stable and non-stable using HP Filter.

    Description:
        The Hodrick-Prescott filter avoids the assumption of a strictly linear
        trend by solving a penalised least squares problem:

            min_tau  sum((y_t - tau_t)^2) + lambda * sum((delta^2 tau_t)^2)

        where delta^2 tau_t = (tau_{t+1} - tau_t) - (tau_t - tau_{t-1})
        is the second difference (convexity) of the trend.

        - Low lambda  : Trend closely tracks data (high convexity, low smoothness).
        - High lambda : Trend is smooth / near-linear (low convexity, high smoothness).

        The stable floor is derived as the trend minus a downward shock equal to
        the (1 - confidence) percentile of the cycle component distribution.
        This ensures the lower bound captures the worst observed cycle deviations
        rather than relying solely on the standard deviation.

    Args:
        balance (pd.Series) : Monthly deposit balance series, indexed by date.
        lam  (float)        : Smoothing parameter lambda (default: 1600 for monthly data).
        confidence (float)  : Confidence level for lower bound (default: 0.95).

    Returns:
        dict: {
            "trend"       : Series — HP trend component,
            "cycle"       : Series — cyclical component (balance - trend),
            "core"        : Series — lower bound of trend (stable floor),
            "non_core"    : Series — volatile portion,
            "stable_pct"  : float  — stable % of total balance,
            "lambda"      : float  — lambda used
        }
    """

    n = len(balance)
    y = balance.values

    eye = np.eye(n)
    d2 = np.diff(eye, n = 2, axis = 0)
    A = eye + lam * (d2.T @ d2)
    trend = np.linalg.solve(A, y)
    cycle = y - trend

    # Use percentile of cycle to set lower bound (captures worst observed dips)
    cycle_floor = np.percentile(cycle, (1 - confidence) * 100)
    lower = trend + cycle_floor
    lower = np.maximum(lower, 0)

    trend_s = pd.Series(trend, index = balance.index, name = "trend")
    cycle_s = pd.Series(cycle, index = balance.index, name = "cycle")
    core = pd.Series(lower, index = balance.index, name = "core")
    non_core = (balance - core).clip(lower = 0)
    non_core.name = "non_core"

    return {
        "trend": trend_s,
        "cycle": cycle_s,
        "core": core,
        "non_core": non_core,
        "stable_pct": float(core.mean() / balance.mean()),
        "lambda": lam
    }

# GBM (Volatility Model)
def stable_gbm(
    balance: pd.Series,
    confidence: float = 0.99,
    sigma_multiplier: float = 1.5
) -> dict:

    """
    Decompose deposit balance into stable and non-stable using Geometric Brownian Motion.

    Description:
        Assumes deposit balances follow a GBM process where log returns are
        normally distributed:
            r_t = ln(B_t / B_{t-1}) ~ N(mu, sigma^2)

        The worst-case decline at a given confidence level is:
            shock = exp(mu + z * sigma_stressed) - 1

        where z = norm.ppf(1 - confidence) (left tail).

        A stressed sigma is used instead of historical sigma to account for
        potential regime changes and fat tails not captured in the observation
        window. sigma_stressed = sigma * sigma_multiplier.

    Args:
        balance (pd.Series)         : Monthly deposit balance series, indexed by date.
        confidence (float)          : Confidence level for downside shock (default: 0.99).
        sigma_multiplier (float)    : Stress multiplier applied to historical sigma (default: 1.5).

    Returns:
        dict: {
            "log_returns"     : Series — monthly log returns,
            "mu"              : float  — mean log return,
            "sigma"           : float  — historical std dev of log returns,
            "sigma_stressed"  : float  — stressed sigma (sigma * multiplier),
            "shock"           : float  — worst-case 1-period decline (decimal),
            "non_stable_pct"  : float  — abs(shock),
            "stable_pct"      : float  — 1 - non_stable_pct,
            "confidence"      : float  — confidence level used
        }
    """

    log_ret = np.log(balance / balance.shift(1)).dropna()
    mu = float(log_ret.mean())
    sigma = float(log_ret.std())
    sigma_stressed = sigma * sigma_multiplier

    z = stats.norm.ppf(1 - confidence)
    shock = np.exp(mu + z * sigma_stressed) - 1
    non_stable_pct = abs(min(shock, 0))

    return {
        "log_returns": log_ret,
        "mu": mu,
        "sigma": sigma,
        "sigma_stressed": sigma_stressed,
        "shock": shock,
        "non_stable_pct": non_stable_pct,
        "stable_pct": 1 - non_stable_pct,
        "confidence": confidence
    }

# Drawdown Model
def stable_drawdown(
    balance: pd.Series,
    horizon: str = "monthly",
    rolling_window: int = 6
) -> dict:

    """
    Decompose deposit balance into stable and non-stable using Drawdown Analysis.

    Description:
        Monthly drawdown represents the percentage decline in balance from one
        period to the next. Two horizons are supported:
        - monthly : drawdown = (balance_t - balance_{t-1}) / balance_{t-1}
        - yearly  : drawdown = (balance_t - balance_{t-12}) / balance_{t-12}

        Rather than using the single worst point in history (which may be an
        outlier), this function uses the worst rolling-window average drawdown.
        This better captures a sustained stress episode over a contiguous period
        and is more robust for setting a non-stable floor.

        Non-stable % = abs(worst rolling average drawdown) if negative, else 0.
        Stable % = 1 - non-stable %.

    Args:
        balance (pd.Series)     : Monthly deposit balance series, indexed by date.
        horizon (str)           : Drawdown horizon, "monthly" or "yearly" (default: "monthly").
        rolling_window (int)    : Window size for rolling average drawdown (default: 24).

    Returns:
        dict: {
            "drawdown"             : Series — full drawdown series (decimal),
            "rolling_avg_drawdown" : Series — rolling average drawdown,
            "worst_drawdown"       : float  — single worst drawdown observed,
            "worst_rolling_avg"    : float  — worst rolling average drawdown,
            "non_stable_pct"       : float  — based on worst rolling average,
            "stable_pct"           : float  — 1 - non_stable_pct,
            "horizon"              : str    — horizon used
        }
    """

    if horizon == "monthly":
        pct_change = balance.pct_change(periods = 1)
    elif horizon == "yearly":
        pct_change = balance.pct_change(periods = 12)
    else:
        raise ValueError("horizon must be 'monthly' or 'yearly'")

    drawdown = pct_change.dropna()

    # Rolling average captures sustained stress better than single worst point
    rolling_avg = drawdown.rolling(window = rolling_window, min_periods = 1).mean()
    worst_rolling = float(rolling_avg.min())
    worst_single = float(drawdown.min())
    non_stable_pct = abs(min(worst_rolling, 0))

    return {
        "drawdown": drawdown,
        "rolling_avg_drawdown": rolling_avg,
        "worst_drawdown": worst_single,
        "worst_rolling_avg": worst_rolling,
        "non_stable_pct": non_stable_pct,
        "stable_pct": 1 - non_stable_pct,
        "horizon": horizon
    }

def compare_stable_methods(
    ci_result: dict,
    hp_result: dict,
    dd_result: dict,
    gbm_result: dict
) -> pd.DataFrame:

    """
    Summarise stable % estimates across all four decomposition methods.

    Description:
        Combines stable % and non-stable % from each method into a single
        comparison table. Useful for benchmarking and selecting the appropriate
        method based on regulatory requirements or model governance.

    Args:
        ci_result (dict)    : Output from stable_ci_method().
        hp_result (dict)    : Output from stable_hp_filter().
        dd_result (dict)    : Output from stable_drawdown().
        gbm_result (dict)   : Output from stable_gbm().

    Returns:
        pd.DataFrame: Comparison table with columns:
                      method, stable_pct, non_stable_pct.
    """

    rows = [
        {
            "method": "Confidence Interval",
            "stable_pct": ci_result["stable_pct"],
            "non_stable_pct": 1 - ci_result["stable_pct"]
        },
        {
            "method": "HP Filter",
            "stable_pct": hp_result["stable_pct"],
            "non_stable_pct": 1 - hp_result["stable_pct"]
        },
        {
            "method": f"GBM (CI={gbm_result['confidence']:.0%})",
            "stable_pct": gbm_result["stable_pct"],
            "non_stable_pct": gbm_result["non_stable_pct"]
        },
        {
            "method": f"Drawdown ({dd_result['horizon']})",
            "stable_pct": dd_result["stable_pct"],
            "non_stable_pct": dd_result["non_stable_pct"]
        },

    ]

    summary = pd.DataFrame(rows).set_index("method")
    summary = (summary * 100).round(2)
    summary.columns = ["Stable (%)", "Non-Stable (%)"]

    return summary