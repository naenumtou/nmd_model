
import numpy as np
import pandas as pd
import warnings

from scipy.optimize import minimize, differential_evolution

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Weighted Average Life (WAL)
def compute_wal(
    survival_prob: np.ndarray,
    runoff_rates: np.ndarray
) -> float:

    """
    Compute Weighted Average Life (WAL) from a runoff profile.

    Description:
        WAL = sum(t_i * CF_i) / sum(CF_i)
        where CF_i = survival_prob[i] * runoff_rate[i] (balance leaving at month i)
        and t_i = i / 12 (time in years).

    Args:
        survival_prob (np.ndarray) : Survival probability at start of each month.
        runoff_rates (np.ndarray)  : Monthly runoff rates.

    Returns:
        float: Weighted average life in years.
    """

    n = len(runoff_rates)
    months = np.arange(1, n + 1)
    cash_flows = survival_prob[:n] * runoff_rates
    total_cf = cash_flows.sum()
    if total_cf == 0:
        return 0.0
    
    return float(np.sum((months / 12) * cash_flows) / total_cf)

# Build runoff profile
def build_runoff_profile(
    hist_runoff: np.ndarray,
    seed_rate: float,
    n_months: int = 60,
    n_period: int = 24
) -> dict:

    """
    Build a full runoff profile combining historical rates and a seed rate.

    Description:
        - Months 1-12  : Observed historical hazard rates from survival decay model.
        - Months 13+   : Constant seed runoff rate (optimized).

        Survival probability at each month is the cumulative product of
        (1 - runoff_rate) from month 1 to that month.

    Args:
        hist_runoff (np.ndarray)        : Historical monthly runoff rates for months n.
        seed_rate (float)               : Constant runoff rate from month 13 onward.
        n_months (int)                  : Total profile horizon in months (default: 60).
        n_period (int)                  : Total historical months runoff rate (default: 24).

    Returns:
        dict: {
            "runoff_rates"  : np.ndarray — monthly runoff rates,
            "survival_prob" : np.ndarray — survival probability at start of each month,
            "cash_flows"    : np.ndarray — balance leaving each month (% of initial),
            "wal_years"     : float      — weighted average life in years,
            "core_pct"      : float      — surviving balance at end of horizon
        }
    """

    runoff_rates = np.empty(n_months)
    runoff_rates[:n_period] = hist_runoff[:min(n_period, len(hist_runoff))]
    runoff_rates[n_period:] = seed_rate

    survival_prob = np.empty(n_months)
    survival_prob[0] = 1.0
    for t in range(1, n_months):
        survival_prob[t] = survival_prob[t - 1] * (1 - runoff_rates[t - 1])

    cash_flows = survival_prob * runoff_rates
    wal = compute_wal(survival_prob, runoff_rates)
    core_pct = float(survival_prob[-1] * (1 - runoff_rates[-1]))

    return {
        "runoff_rates": runoff_rates,
        "survival_prob": survival_prob,
        "cash_flows": cash_flows,
        "wal_years": wal,
        "core_pct": core_pct
    }

# Find the seed rate
def optimize_seed_rate(
    hist_runoff: np.ndarray,
    max_wal_years: float = 5.0,
    max_core_pct: float = 0.90,
    n_months: int = 60,
    n_period: int = 24
) -> dict:

    """
    Find the seed runoff rate that maximises WAL subject to IRRBB constraints.

    Description:
        Maximise   WAL(seed_rate)
        Subject to WAL  <= max_wal_years   (BCBS IRRBB behavioural maturity cap)
                   core % <= max_core_pct  (regulatory deposit cap)
                   seed_rate >= 0

        A lower seed rate → slower long-run runoff → longer WAL.
        Uses differential_evolution for global optimality.

    Args:
        hist_runoff_12m (np.ndarray)    : Historical runoff rates for months 1-12.
        max_wal_years (float)           : Maximum WAL in years (default: 5.0).
        max_core_pct (float)            : Maximum core % at end of horizon (default: 0.90).
        n_months (int)                  : Horizon in months (default: 60).
        n_period (int)                  : Total historical months runoff rate (default: 24).

    Returns:
        dict: {
            "seed_rate" : float — optimal seed runoff rate,
            "wal_years" : float — resulting WAL,
            "core_pct"  : float — surviving core % at end of horizon,
            "profile"   : dict  — full runoff profile,
            "converged" : bool  — whether constraints were satisfied
        }
    """

    def neg_wal(seed):
        profile = build_runoff_profile(hist_runoff, seed[0], n_months, n_period)
        wal = profile["wal_years"]
        core = profile["core_pct"]
        penalty = 0.0
        if wal > max_wal_years:
            penalty += 1000 * (wal - max_wal_years) ** 2
        if core > max_core_pct:
            penalty += 1000 * (core - max_core_pct) ** 2
        return -wal + penalty

    result = differential_evolution(
        neg_wal, bounds = [(0.001, 0.30)], seed = 42, tol = 1e-8, maxiter = 500
    )

    best_seed = float(result.x[0])
    best_profile = build_runoff_profile(hist_runoff, best_seed, n_months, n_period)
    converged = (
        best_profile["wal_years"] <= max_wal_years
        and best_profile["core_pct"] <= max_core_pct
    )

    return {
        "seed_rate": best_seed,
        "wal_years": best_profile["wal_years"],
        "core_pct": best_profile["core_pct"],
        "profile": best_profile,
        "converged": converged
    }

# IRRBB Repricing buckets
def build_irrbb_buckets(
    profile: dict,
    core_balance_mb: float
) -> pd.DataFrame:

    """
    Translate monthly runoff profile into IRRBB quarterly repricing buckets.

    Description:
        Allocates core deposit cash flows into standard IRRBB time buckets.
        Cash flow at month t = survival_prob[t] * runoff_rate[t] * core_balance_mb.
        Residual surviving balance at end of 60-month horizon goes to 5Y+ bucket.

    Args:
        profile (dict)          : Output from build_runoff_profile().
        core_balance_mb (float) : Core balance in MB.

    Returns:
        pd.DataFrame: IRRBB bucket table with balance_mb and pct_of_core.
    """

    cf_mb = profile["cash_flows"] * core_balance_mb
    bucket_defs = [
        ("1-3M",  1,  3),
        ("3-6M",  4,  6),
        ("6-12M", 7, 12),
        ("1-2Y", 13, 24),
        ("2-3Y", 25, 36),
        ("3-4Y", 37, 48),
        ("4-5Y", 49, 60)
    ]
    rows = []
    total_allocated = 0.0
    for label, m_start, m_end in bucket_defs:
        idx = slice(m_start - 1, min(m_end, len(cf_mb)))
        bucket_mb = float(cf_mb[idx].sum())
        total_allocated += bucket_mb
        rows.append(
            {
                "bucket": label, "months": f"{m_start}-{m_end}",
                "balance_mb": round(bucket_mb, 2),
                "pct_of_core": bucket_mb / core_balance_mb
            }
        )

    residual_mb = max(core_balance_mb - total_allocated, 0)
    rows.append(
        {
            "bucket": "5Y+", "months": "61+",
            "balance_mb": round(residual_mb, 2),
            "pct_of_core": residual_mb / core_balance_mb
        }
    )

    result = pd.DataFrame(rows).set_index("bucket")
    result["pct_of_core"] = (result["pct_of_core"] * 100).round(2)
    
    return result

# Synthetic yield curve
def build_yield_curve(
    market_rate: pd.Series,
    tenors_years: list
) -> pd.DataFrame:

    """
    Construct a synthetic yield curve with realistic term structure dynamics.

    Description:
        Each tenor has its own AR(1) dynamics and idiosyncratic noise,
        producing a yield curve where tenors behave distinctly enough for
        the replicating portfolio optimizer to differentiate between them.

        Key design choices:
            - phi (persistence): short tenors are reactive (low phi), long tenors
              are sticky (high phi). This mimics how short-end rates track the
              central bank more closely than long-end rates.
            - term_premium: calibrated to produce a realistic upward-sloping
              curve with a positive slope from 3M to 10Y.
            - idiosyncratic vol: higher at the short end (monetary policy noise),
              lower at the long end (inflation expectations are sticky).

        These differences in dynamics create sufficient heterogeneity across
        tenors so that the optimizer can assign meaningful weights.

    Args:
        market_rate (pd.Series)  : 10Y market rate series, indexed by date.
        tenors_years (list)      : Tenor lengths in years (e.g., [0.25, 0.5, 1, 2, 3, 5, 10]).

    Returns:
        pd.DataFrame: Yield curve, shape (n_months, n_tenors).
    """

    rng = np.random.default_rng(99)
    r10 = market_rate.values
    n = len(r10)

    # Short rates react fast to market, long rates are sticky
    phi_map = {
        0.25: 0.30,
        0.5: 0.45,
        1: 0.58,
        2: 0.70,
        3: 0.78,
        5: 0.86,
        10: 0.93
    }
    
    # Upward sloping term premium
    tp_map  = {
        0.25: -0.018,
        0.5: -0.014,
        1: -0.010,
        2: -0.006,
        3: -0.003,
        5: 0.002,
        10: 0.006
    }
    
    # Short end more volatile (monetary policy noise)
    vol_map = {
        0.25: 0.0025,
        0.5: 0.0020,
        1: 0.0016,
        2: 0.0012,
        3: 0.0010,
        5: 0.0008,
        10: 0.0005
    }

    def tenor_label(t):
        return f"{int(t * 12)}M" if t < 1 else f"{int(t)}Y"

    curve = {}
    for t in tenors_years:
        phi = phi_map[t]
        tp  = tp_map[t]
        vol = vol_map[t]
        idio = rng.normal(0, vol, n)
        series = np.empty(n)
        long_run = r10 + tp
        series[0] = long_run[0]
        for i in range(1, n):
            series[i] = phi * series[i-1] + (1 - phi) * long_run[i] + idio[i]
        curve[tenor_label(t)] = np.clip(series, 0.001, None)

    return pd.DataFrame(curve, index = market_rate.index)

# Replicating portfolio
def static_replicating_portfolio(
    deposit_rate: pd.Series,
    market_rate: pd.Series,
    core_balance_mb: float,
    tenors_years: list = None,
    max_wal_years: float = 5.0,
    max_weight_per_tenor: float = 0.50
) -> dict:

    """
    Find optimal weights for a static replicating portfolio of fixed-maturity instruments.

    Description:
        The objective is to find weights w_i and a margin m such that:
        deposit_rate_t ≈ Σ(w_i x yield_curve_t(tenor_i)) - margin

        This reflects the economic reality that a bank earns the portfolio
        yield on its core deposits and pays the deposit rate to clients,
        retaining the margin as net interest income.

        Objective:
            Minimise tracking_error = std(portfolio_return - margin - deposit_rate)

        Constraints:
            sum(w_i) = 1                    (fully invested)
            0 <= w_i <= max_weight_per_tenor (long-only, capped per tenor)
            WAL = sum(w_i * tenor_i) <= max_wal_years  (IRRBB maturity cap)
            margin >= 0                     (non-negative bank margin)

        The WAL constraint and per-tenor cap prevent degenerate solutions
        where all weight concentrates in the longest tenor.

    Args:
        deposit_rate          (pd.Series) : Client deposit rate (Target to replicate).
        market_rate           (pd.Series) : 10Y market rate (Used to build yield curve).
        core_balance_mb       (float)     : Core balance in MB (For notional allocation).
        tenors_years          (list)      : Tenor list in years (Default: [0.25, 0.5, 1, 2, 3, 5, 10]).
        max_wal_years         (float)     : WAL cap in years (Default: 5.0).
        max_weight_per_tenor  (float)     : Max weight per tenor (Default: 0.50 = 50%).

    Returns:
        dict: {
            "weights"         : Series — optimal weights per tenor,
            "margin"          : float  — bank margin (portfolio yield - deposit rate),
            "notional_mb"     : Series — MB allocated per tenor,
            "tracking_error"  : float  — std of (portfolio_return - margin - deposit_rate),
            "portfolio_return": Series — fitted portfolio return series,
            "yield_curve"     : DataFrame — synthetic yield curve per tenor,
            "wal_years"       : float  — weighted average tenor
        }
    """

    if tenors_years is None:
        tenors_years = [0.25, 0.5, 1, 2, 3, 5, 10]

    yield_curve = build_yield_curve(market_rate, tenors_years)
    combined = pd.concat([deposit_rate, yield_curve], axis = 1).dropna()
    target = combined.iloc[:, 0].values
    X = combined.iloc[:, 1:].values
    n_tenors = X.shape[1]
    tenors_arr = np.array(tenors_years)

    # Params: [w_0 ... w_{n-1}, margin]
    def objective(params):
        w = params[:n_tenors]
        margin = params[n_tenors]
        portfolio = X @ w - margin
        te = float(np.std(portfolio - target))
        wal = float(np.dot(w, tenors_arr))
        wal_penalty = max(0, wal - max_wal_years) * 20
        return te + wal_penalty

    constraints = [
        {
            "type": "eq",
            "fun": lambda p: np.sum(p[:n_tenors]) - 1
        }
    ]
    
    bounds = [(0.0, max_weight_per_tenor)] * n_tenors + [(0.0, 0.10)]
    p0 = np.array([1 / n_tenors] * n_tenors + [0.03])

    result = minimize(
        objective, x0 = p0, method = "SLSQP", bounds = bounds,
        constraints = constraints, options = {"ftol": 1e-12, "maxiter": 3000}
    )

    w_opt = result.x[:n_tenors]
    margin_opt = float(result.x[n_tenors])
    tenor_labels = yield_curve.columns.tolist()
    portfolio_ret = pd.Series(
        X @ w_opt - margin_opt,
        index = combined.index,
        name = "portfolio_return"
    )
    wal = float(np.dot(w_opt, tenors_arr))

    return {
        "weights": pd.Series(w_opt, index = tenor_labels, name = "weight"),
        "margin": margin_opt,
        "notional_mb": pd.Series(w_opt * core_balance_mb, index = tenor_labels, name = "notional_mb"),
        "tracking_error": float(np.std(X @ w_opt - margin_opt - target)),
        "portfolio_return": portfolio_ret,
        "yield_curve": yield_curve,
        "wal_years": wal
    }