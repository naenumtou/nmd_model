
import numpy as np
import pandas as pd
import warnings

from scipy import stats

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Cohort simulation from month series
def simulate_cohorts(
    balance: pd.Series,
    n_cohorts: int,
    seed: int = 42
) -> pd.DataFrame:

    """
    Simulate a cohort balance matrix from an aggregate balance series.

    Description:
        In practice, each cohort represents a group of accounts opened in the
        same month. Their balances are tracked over time to observe runoff.

        Since we do not have account-level data, we decompose the aggregate
        balance into cohorts by:
        - Assigning a random share of new inflows to each cohort at origination.
        - Applying cohort-specific decay rates that vary slightly around the
          aggregate trend, reflecting heterogeneity in depositor behaviour.

        The resulting matrix has shape (n_months, n_cohorts), where entry (t, c)
        is the surviving balance of cohort c at time t.

    Args:
        balance (pd.Series)     : Aggregate monthly deposit balance (MB), indexed by date.
        n_cohorts (int)         : Number of cohorts to simulate.
        seed (int)              : Random seed for reproducibility.

    Returns:
        pd.DataFrame: Cohort balance matrix shape (n_months, n_cohorts).
                      Columns are cohort origination dates.
                      Entry is NaN for periods before the cohort originated.
    """

    rng = np.random.default_rng(seed)
    n_months = len(balance)
    dates = balance.index
    balances = balance.values

    # Cohort origination months evenly spaced across the observation window
    cohort_idx = np.linspace(0, n_months - 2, n_cohorts, dtype = int)
    matrix = pd.DataFrame(np.nan, index = dates, columns = dates[cohort_idx])

    for c, orig in enumerate(cohort_idx):

        share = rng.uniform(0.01, 0.04)
        bal_0 = balances[orig] * share

        # Cohort-specific monthly decay rate (heterogeneous around 2-6%)
        base_decay = rng.uniform(0.02, 0.06)

        cohort_bal = np.empty(n_months - orig)
        cohort_bal[0] = bal_0

        for t in range(1, len(cohort_bal)):
            decay_t = base_decay + rng.normal(0, 0.005)
            decay_t = np.clip(decay_t, 0.001, 0.20)
            cohort_bal[t] = cohort_bal[t - 1] * (1 - decay_t)

        matrix.iloc[orig:, c] = cohort_bal

    return matrix

# Runoff rates
def compute_cohort_runoff_rates(
    cohort_matrix: pd.DataFrame
) -> pd.DataFrame:

    """
    Compute month-over-month runoff rates for each cohort.

    Description:
        Runoff rate at time t for cohort c is defined as:
            runoff_rate(t, c) = (Balance(t-1, c) - Balance(t, c)) / Balance(t-1, c)
                              = 1 - (Balance(t, c) / Balance(t-1, c))

        A positive value indicates balance decline (runoff).
        A negative value indicates balance growth (net inflow to cohort).

        NaN is returned for the origination period (no prior balance exists)
        and for any period where the prior balance is zero or NaN.

    Args:
        cohort_matrix (pd.DataFrame): Cohort balance matrix from simulate_cohorts().
                                      Shape (n_months, n_cohorts).

    Returns:
        pd.DataFrame: Runoff rate matrix, same shape as cohort_matrix.
                      Values represent decimal runoff rates (e.g., 0.03 = 3%).
    """

    runoff = 1 - (cohort_matrix / cohort_matrix.shift(1))

    return runoff

# Hazard rate
def compute_snapshot_hazard_rates(
    runoff_rates: pd.DataFrame
) -> pd.Series:

    """
    Compute the cross-cohort average runoff rate for each snapshot (time period).

    Description:
        For each snapshot t, we take the average runoff rate across all cohorts
        that are alive at time t. This produces a single discrete hazard rate
        per period that captures the aggregate deposit decay behaviour.

        These snapshot hazard rates are then used as the dependent variable
        in the MEV regression (Step 4b).

    Args:
        runoff_rates (pd.DataFrame): Runoff rate matrix from compute_cohort_runoff_rates().

    Returns:
        pd.Series: Monthly snapshot hazard rates, indexed by date.
                   Name: "hazard_rate".
    """

    hazard = runoff_rates.mean(axis=1, skipna=True)
    hazard.name = "hazard_rate"

    return hazard

# Worst case
def worst_case_runoff(
    hazard_rates: pd.Series,
    percentile: float = 95.0
) -> dict:

    """
    Derive the worst-case runoff rate from the snapshot hazard rate time series.

    Description:
        The worst-case percentile approach is used when MEV data is unavailable
        or as a conservative stress assumption. It takes the upper tail of the
        observed hazard rate distribution as the stressed runoff rate.

        Applied in ILAAP / liquidity stress testing contexts where a single
        conservative runoff rate is needed for the full horizon.

    Args:
        hazard_rates (pd.Series) : Snapshot hazard rate series.
        percentile  (float)      : Percentile to use as worst case (default: 95th).

    Returns:
        dict: {
            "worst_case_rate" : float  — worst-case runoff rate (decimal),
            "percentile"      : float  — percentile used,
            "mean_rate"       : float  — mean hazard rate,
            "std_rate"        : float  — std dev of hazard rate,
            "distribution"    : Series — full clean series (for plotting)
        }
    """

    clean = hazard_rates.dropna()

    return {
        "worst_case_rate": float(np.percentile(clean, percentile)),
        "percentile": percentile,
        "mean_rate": float(clean.mean()),
        "std_rate": float(clean.std()),
        "distribution": clean
    }

# Regression of hazard rate
def regress_runoff_on_mev(
    hazard_rates: pd.Series,
    mev_df: pd.DataFrame,
    mev_cols: list
) -> dict:

    """
    Regress snapshot hazard rates on macro economic variables (MEVs).

    Description:
        Models the relationship between deposit runoff rates and macro conditions:
            hazard_rate_t = alpha + beta_1 * MEV_1_t + beta_2 * MEV_2_t + eps_t

        Allows the bank to project runoff rates under different macro scenarios
        (e.g., rising rates, rising unemployment).

        OLS is used with an intercept. The function computes:
        - Coefficients, t-statistics, p-values
        - R-squared and adjusted R-squared
        - Fitted values and residuals

    Args:
        hazard_rates (pd.Series)    : Snapshot hazard rates (dependent variable).
        mev_df (pd.DataFrame)       : DataFrame containing MEV series, indexed by date.
        mev_cols (list)             : Column names in mev_df to use as regressors.

    Returns:
        dict: {
            "coefficients" : Series — estimated coefficients (including intercept),
            "tstat"        : Series — t-statistics,
            "pvalue"       : Series — p-values,
            "r2"           : float  — R-squared,
            "adj_r2"       : float  — adjusted R-squared,
            "fitted"       : Series — fitted values,
            "residuals"    : Series — OLS residuals
        }
    """

    combined = pd.concat([hazard_rates, mev_df[mev_cols]], axis=1).dropna()
    y = combined.iloc[:, 0].values
    X_raw = combined[mev_cols].values
    X = np.column_stack([np.ones(len(y)), X_raw])

    beta = np.linalg.lstsq(X, y, rcond = None)[0]
    fitted = X @ beta
    residuals = y - fitted

    n, k = X.shape
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    sigma2 = sse / (n - k)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
    tstat = beta / se
    pvalue = 2 * (1 - stats.t.cdf(np.abs(tstat), df = n - k))

    col_names = ["intercept"] + mev_cols

    return {
        "coefficients": pd.Series(beta, index = col_names, name = "coefficient"),
        "tstat": pd.Series(tstat, index = col_names, name = "t_stat"),
        "pvalue": pd.Series(pvalue, index = col_names, name = "p_value"),
        "r2": r2,
        "adj_r2": adj_r2,
        "fitted": pd.Series(fitted, index = combined.index, name = "fitted"),
        "residuals": pd.Series(residuals, index = combined.index, name = "residuals")
    }