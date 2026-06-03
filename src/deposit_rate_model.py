
import numpy as np
import pandas as pd
import warnings

from scipy import stats
from scipy.optimize import minimize

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Beta regression
def deposit_rate_linear(
    deposit_rate: pd.Series,
    market_rate: pd.Series
) -> dict:

    """
    Estimate deposit rate pass-through using linear OLS regression.

    Description:
        Regresses client deposit rates on market rates (level-based model):
            d_t = alpha + beta * r_t + eps_t

        The beta coefficient measures the long-run pass-through of market rate
        changes to deposit rates. For IRRBB purposes:
        - beta fraction of deposits reprices immediately (overnight bucket).
        - (1 - beta) fraction is behaviourally sticky (longer bucket).

        For example, beta = 0.25 implies that a 100bps (1%) rise in market rates
        results in only a 25bps (0.25%) rise in deposit rates. From a behavioural
        perspective, 25% of deposits are treated as rate-sensitive.

    Args:
        deposit_rate (pd.Series)    : Client deposit rate series (decimal).
        market_rate (pd.Series)     : Market / risk-free rate series (decimal).

    Returns:
        dict: {
            "alpha"       : float  — intercept,
            "beta"        : float  — pass-through coefficient,
            "r2"          : float  — R-squared,
            "tstat_beta"  : float  — t-statistic of beta,
            "pvalue_beta" : float  — p-value of beta,
            "fitted"      : Series — fitted deposit rates,
            "residuals"   : Series — OLS residuals
        }
    """

    combined = pd.concat([deposit_rate, market_rate], axis = 1).dropna()
    y = combined.iloc[:, 0].values
    x = combined.iloc[:, 1].values
    n = len(y)

    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond = None)[0]
    fitted = X @ beta
    residuals = y - fitted

    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst

    sigma2 = sse / (n - 2)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
    tstat = beta / se
    pvalue = 2 * (1 - stats.t.cdf(np.abs(tstat), df = n - 2))

    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]),
        "r2": r2,
        "tstat_beta": float(tstat[1]),
        "pvalue_beta": float(pvalue[1]),
        "fitted": pd.Series(fitted, index = combined.index, name = "fitted"),
        "residuals": pd.Series(residuals, index = combined.index, name = "residuals")
    }

# Threshold model
def deposit_rate_threshold(
    deposit_rate: pd.Series,
    market_rate: pd.Series,
    beta_ols: float,
    init_params: list = None
) -> dict:

    """
    Estimate asymmetric deposit rate adjustment using a threshold model.

    Description:
        Captures the empirical observation that banks adjust deposit rates
        asymmetrically — raising rates slowly and cutting them quickly.

        Model equations:
            d_eq_t = beta * r_t - g                      (equilibrium rate)
            I_t    = 1 if d_eq_t > d_{t-1} else 0        (indicator: rates need to rise)
            d_t    = d_{t-1} + (lambda+ * I_t + lambda- * (1 - I_t)) * (d_eq_t - d_{t-1}) + eps_t

        Parameters estimated via minimising sum of squared residuals:
            theta = [b, g, lambda+, lambda-]

        Constraints:
            g >= 0         : non-negative spread
            0 < lambda+ < 1: upward adjustment speed
            0 < lambda- < 1: downward adjustment speed

    Args:
        deposit_rate (pd.Series)    : Client deposit rate series (decimal).
        market_rate (pd.Series)     : Market / risk-free rate series (decimal).
        beta_ols (float)            : Beta estimated from OLS Model (Beta regression).
        init_params  (list)         : Initial parameter guess [g, lambda+, lambda-]. Default is [0.005, 0.15, 0.45].

    Returns:
        dict: {
            "b"          : float  — long-run pass-through,
            "g"          : float  — bank spread (margin retained),
            "lambda_up"  : float  — upward adjustment speed,
            "lambda_down": float  — downward adjustment speed,
            "fitted"     : Series — fitted deposit rates,
            "residuals"  : Series — model residuals,
            "sse"        : float  — sum of squared errors,
            "converged"  : bool   — optimisation convergence status
        }
    """

    if init_params is None:
        init_params = [0.005, 0.15, 0.45]

    combined = pd.concat([deposit_rate, market_rate], axis = 1).dropna()
    d = combined.iloc[:, 0].values
    r = combined.iloc[:, 1].values
    n = len(d)

    def simulate_threshold(params):
        g, lam_up, lam_down = params
        b = beta_ols
        d_hat = np.empty(n)
        d_hat[0] = d[0]
        for t in range(1, n):
            d_eq = b * r[t] - g
            gap = d_eq - d_hat[t - 1]
            lam = lam_up if gap > 0 else lam_down
            d_hat[t] = d_hat[t - 1] + lam * gap
            d_hat[t] = max(d_hat[t], 0)
        return d_hat

    def objective(params):
        d_hat = simulate_threshold(params)
        return np.sum((d - d_hat) ** 2)

    bounds = [(0.0, 0.05), (0.01, 0.99), (0.01, 0.99)]
    result = minimize(
        objective,
        x0 = init_params,
        bounds = bounds,
        method = "L-BFGS-B"
    )

    g_hat, lam_up_hat, lam_down_hat = result.x
    fitted = simulate_threshold(result.x)
    residuals = d - fitted

    return {
        "beta": float(beta_ols),
        "g": float(g_hat),
        "lambda_up": float(lam_up_hat),
        "lambda_down": float(lam_down_hat),
        "fitted": pd.Series(fitted, index = combined.index, name = "fitted"),
        "residuals": pd.Series(residuals, index = combined.index, name = "residuals"),
        "sse": float(result.fun),
        "converged": bool(result.success)
    }

# Jarrow-Van Deventer model
def deposit_rate_jvd(
    deposit_rate: pd.Series,
    market_rate: pd.Series
) -> dict:

    """
    Estimate deposit rate dynamics using the Jarrow-Van Deventer (JVD) framework.

    Description:
        Unlike level-based models, JVD models changes in deposit rates as a
        function of changes in market rates and a time trend:

            Delta_d_t = beta_0 + beta_1 * t + beta_2 * Delta_r_t + eps_t

        where:
            - beta_0 : Autonomous drift in deposit rates
            - beta_1 : Secular time trend (e.g., structural repricing over time)
            - beta_2 : Immediate pass-through intensity (rate adjustment when bank reprices)

        JVD incorporates stickiness naturally since banks may not reprice every
        period, so the model fits changes only when they occur. This makes it
        more suitable for NII forecasting than for direct IRRBB bucketing,
        where the level-based beta is preferred.

    Args:
        deposit_rate (pd.Series) : Client deposit rate series (decimal).
        market_rate (pd.Series)  : Market / risk-free rate series (decimal).

    Returns:
        dict: {
            "beta_0"      : float  — intercept (autonomous drift),
            "beta_1"      : float  — time trend coefficient,
            "beta_2"      : float  — change pass-through coefficient,
            "r2"          : float  — R-squared on change regression,
            "tstat"       : Series — t-statistics for all coefficients,
            "pvalue"      : Series — p-values for all coefficients,
            "fitted"      : Series — fitted changes in deposit rate,
            "fitted_level": Series — reconstructed deposit rate level,
            "residuals"   : Series — residuals on change regression
        }
    """

    delta_d = deposit_rate.diff().dropna()
    delta_r = market_rate.diff().dropna()

    combined = pd.concat([delta_d, delta_r], axis = 1).dropna()
    combined.columns = ["delta_d", "delta_r"]
    n = len(combined)
    t = np.arange(n)

    y = combined["delta_d"].values
    X = np.column_stack([np.ones(n), t, combined["delta_r"].values])

    beta = np.linalg.lstsq(X, y, rcond = None)[0]
    fitted_changes = X @ beta
    residuals = y - fitted_changes

    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst

    sigma2 = sse / (n - 3)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
    tstat = beta / se
    pvalue = 2 * (1 - stats.t.cdf(np.abs(tstat), df = n - 3))

    col_names = ["beta_0", "beta_1", "beta_2"]

    fitted_level = deposit_rate.iloc[0] + np.concatenate([[0], np.cumsum(fitted_changes)])
    fitted_level = pd.Series(
        fitted_level[:len(deposit_rate)],
        index = deposit_rate.index,
        name = "fitted_level"
    )

    return {
        "beta_0": float(beta[0]),
        "beta_1": float(beta[1]),
        "beta_2": float(beta[2]),
        "r2": r2,
        "tstat": pd.Series(tstat, index = col_names, name = "t_stat"),
        "pvalue": pd.Series(pvalue, index = col_names, name = "p_value"),
        "fitted": pd.Series(fitted_changes, index = combined.index, name = "fitted_changes"),
        "fitted_level": fitted_level,
        "residuals": pd.Series(residuals, index = combined.index, name = "residuals")
    }