
import numpy as np
import pandas as pd
import warnings

from scipy.special import softmax

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Generate a mean-reverting
def generate_ar1(
    n: int,
    mu: float,
    phi: float,
    sigma: float,
    x0: float,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate a mean-reverting AR(1) time series.

    Description:
        Simulates the process: x_t = mu + phi * (x_{t-1} - mu) + sigma * eps_t
        where eps_t ~ N(0, 1). This is a discrete-time Ornstein-Uhlenbeck
        process commonly used to model interest rates and spreads.

    Args:
        n (int)                     : Number of periods to simulate.
        mu (float)                  : Long-run mean (level of mean reversion).
        phi (float)                 : Persistence parameter (0 < phi < 1).
        sigma (float)               : Volatility of innovations.
        x0 (float)                  : Initial value at t=0.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        np.ndarray: Simulated time series of length n.
    """

    x = np.empty(n)
    x[0] = x0
    eps = rng.standard_normal(n)

    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * eps[t]

    return x

# Generate market rate and repo rate
def generate_market_rates(
    n: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:

    """
    Generate correlated market rate and repo rate series.

    Description:
        Both rates follow AR(1) mean-reverting processes and share a common
        innovation component, introducing positive correlation between them.
        - market_rate: 10-year bond yield proxy.
        - repo_rate  : Policy / overnight rate proxy (lower level, tighter range).

    Args:
        n (int)                     : Number of periods.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        tuple[np.ndarray, np.ndarray]: (market_rate, repo_rate), both in decimal
                                       (e.g., 0.05 = 5%).
    """

    # Shared innovation drives correlation between the two rates
    common_shock = rng.standard_normal(n)
    idio_market = rng.standard_normal(n)
    idio_repo = rng.standard_normal(n)

    corr = 0.80 #Target correlation
    eps_mkt  = corr * common_shock + np.sqrt(1 - corr ** 2) * idio_market
    eps_repo = corr * common_shock + np.sqrt(1 - corr ** 2) * idio_repo

    # Market rate: 10Y bond yield
    mkt = np.empty(n)
    mkt[0] = 0.055   # start at 5.5%
    for t in range(1, n):
        mkt[t] = 0.05 + 0.85 * (mkt[t - 1] - 0.05) + 0.004 * eps_mkt[t]
    mkt = np.clip(mkt, 0.005, 0.15)

    # Repo rate: lower level, less volatile
    repo = np.empty(n)
    repo[0] = 0.025
    for t in range(1, n):
        repo[t] = 0.025 + 0.80 * (repo[t - 1] - 0.025) + 0.002 * eps_repo[t]
    repo = np.clip(repo, 0.001, 0.10)

    return mkt, repo

# Generate deposit rate
def generate_deposit_rate(
    market_rate: np.ndarray,
    beta: float,
    lambda_up: float,
    lambda_down: float,
    spread: float,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate deposit rate with asymmetric adjustment speed (threshold structure).

    Description:
        Deposit rate follows a partial adjustment model toward an equilibrium
        target defined by the market rate. Adjustment is asymmetric:
        - Banks raise deposit rates slowly (lambda_up is small).
        - Banks lower deposit rates quickly (lambda_down is large).

        Equilibrium: d_eq_t = beta * market_rate_t - spread
        Update rule: d_t = d_{t-1} + lambda * (d_eq_t - d_{t-1}) + eps

    Args:
        market_rate (np.ndarray)    : Market / risk-free rate series.
        beta (float)                : Long-run pass-through coefficient.
        lambda_up (float)           : Adjustment speed when target > current.
        lambda_down (float)         : Adjustment speed when target < current.
        spread (float)              : Bank margin retained (subtracted from target).
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        np.ndarray: Simulated deposit rate series.
    """

    n = len(market_rate)
    d = np.empty(n)
    d[0] = beta * market_rate[0] - spread
    eps = rng.normal(0, 0.0005, n) #Small idiosyncratic noise

    for t in range(1, n):
        d_eq = beta * market_rate[t] - spread
        gap = d_eq - d[t - 1]
        lam = lambda_up if gap > 0 else lambda_down
        d[t] = d[t - 1] + lam * gap + eps[t]

    return np.clip(d, 0.001, None)

# Generate monthly deposit
def generate_balance(
    n: int,
    market_rate: np.ndarray,
    deposit_rate: np.ndarray,
    cds_spread: np.ndarray,
    initial_balance: float,
    trend_monthly: float,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate monthly deposit balance series.

    Description:
        Balance evolves as a log-linear process with:
        - A deterministic upward trend (organic growth).
        - Negative sensitivity to opportunity cost (market_rate - deposit_rate):
          when the gap widens, depositors move funds to higher-yielding alternatives.
        - Negative sensitivity to bank stress (high CDS spread):
          a dummy fires when CDS > 100 bps, triggering a deposit outflow shock.
        - Log-normal idiosyncratic noise.

        The underlying model mirrors the economic theory deposit volume equation:
        Delta ln(D_t) = phi + lambda*(d_t - r_t) + theta*1(CDS>100bps) + eps_t

    Args:
        n (int)                     : Number of periods.
        market_rate (np.ndarray)    : Market / risk-free rate series.
        deposit_rate (np.ndarray)   : Client deposit rate series.
        cds_spread (np.ndarray)     : CDS spread series (in bps).
        initial_balance (float)     : Starting balance (MB).
        trend_monthly (float)       : Deterministic monthly growth rate.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        np.ndarray: Monthly deposit balance series (MB).
    """

    # Model coefficients
    phi = trend_monthly #Baseline growth
    lam = -0.20 #Opportunity cost sensitivity
    theta = -0.025 #CDS stress shock
    sigma = 0.018  #Idiosyncratic noise

    opp_cost  = market_rate - deposit_rate
    cds_dummy = (cds_spread > 100).astype(float) #Stress indicator

    # Seasonal component: Q1 outflow (Jan-Mar), Q4 inflow (Oct-Dec)
    month_idx = np.arange(n) % 12
    seasonal = np.where(
        month_idx < 3,
        -0.012,
        np.where(
            month_idx >= 9,
            0.008, 0.0
        )
    )
    
    log_d = np.empty(n)
    log_d[0] = np.log(initial_balance)
    eps = rng.normal(0, sigma, n)

    for t in range(1, n):
        delta = phi + lam * opp_cost[t] + theta * cds_dummy[t] + seasonal[t] + eps[t]
        log_d[t] = log_d[t - 1] + delta

    return np.exp(log_d)

# Generate unemployment rate (MEV)
def generate_unemployment(
    n: int,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate unemployment rate as an independent AR(1) MEV series.

    Description:
        Unemployment follows a slow mean-reverting process with low volatility,
        consistent with typical macroeconomic data. Used as an exogenous
        macro-economic variable (MEV) in the Survival/Decay regression model.

    Args:
        n (int)                     : Number of periods.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        np.ndarray: Monthly unemployment rate series (decimal).
    """

    return generate_ar1(
        n = n, mu = 0.04, phi = 0.92, sigma = 0.003, x0 = 0.045, rng = rng
    ).clip(0.01, 0.20)

# Generate CDS spread series
def generate_cds_spread(
    n: int,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate CDS spread series with stress spikes.

    Description:
        CDS spread follows a log-AR(1) process to ensure non-negativity.
        Occasional stress events are injected by adding a random spike at a
        randomly chosen period, followed by mean reversion.
        Used in the Economic Theory model and as a deposit volume driver.

    Args:
        n (int)                   : Number of periods.
        rng (np.random.Generator) : NumPy random generator instance.

    Returns:
        np.ndarray: CDS spread series in basis points (bps).
    """

    log_cds = generate_ar1(
        n = n, mu = np.log(60), phi = 0.88, sigma = 0.12, x0 = np.log(55), rng = rng
    )

    # Inject one stress episode
    spike_start = rng.integers(40, 90)
    spike_len = rng.integers(6, 15)
    spike_idx = slice(spike_start, spike_start + spike_len)
    log_cds[spike_idx] += rng.uniform(0.6, 1.2, spike_len)

    return np.exp(log_cds).clip(10, 600)

# Generate client wealth index
def generate_wealth_and_equity(
    n: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Generate client wealth index, equity log-returns, and equity implied volatility.

    Description:
        - Wealth: grows with a positive trend, partially driven by equity returns
          (reflecting the typical composition of household wealth).
        - Equity return: random walk with drift (log-normal price process).
        - Equity vol: mean-reverting process negatively correlated with equity
          returns (leverage effect — volatility rises when prices fall).

        These series are used as explanatory variables in the Wealth Allocation
        (Bernoulli-Beta mixture) model.

    Args:
        n (int)                     : Number of periods.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (wealth_index, equity_return, equity_vol)
    """

    # Equity log-returns
    eq_ret = rng.normal(0.005, 0.035, n) #~6% annual drift

    # Equity implied vol (leverage effect: high vol when returns are low)
    eq_vol = np.empty(n)
    eq_vol[0] = 0.18
    vol_eps = rng.standard_normal(n)
    for t in range(1, n):
        eq_vol[t] = (
            0.18
            + 0.75 * (eq_vol[t - 1] - 0.18)
            - 0.10 * eq_ret[t] #Leverage effect
            + 0.015 * vol_eps[t]
        )
    eq_vol = np.clip(eq_vol, 0.05, 0.80)

    # Wealth index: trend + equity component
    wealth = np.empty(n)
    wealth[0] = 1000.0 #Index base = 1,000
    wealth_eps = rng.normal(0, 0.005, n)
    for t in range(1, n):
        delta_w = 0.003 + 0.40 * eq_ret[t] + wealth_eps[t]
        wealth[t] = wealth[t - 1] * np.exp(delta_w)

    return wealth, eq_ret, eq_vol

# Combined all series
def generate_nmd_dataset(
    n_months: int,
    start_date: str,
    rng: np.random.Generator
) -> pd.DataFrame:

    """
    Generate a complete synthetic NMD dataset for all downstream models.

    Description:
        Orchestrates all individual generator functions and assembles the
        results into a single DataFrame indexed by month-end dates.
        The relationships embedded in the data are:
        - deposit_rate partially tracks market_rate (beta = 0.30)
        - balance responds to opportunity cost and CDS stress
        - repo_rate and market_rate are positively correlated
        - equity_vol is negatively correlated with equity_return
        - wealth is driven by equity returns and a trend

    Args:
        n_months (int)              : Number of monthly observations.
        start_date (str)            : Start date in "YYYY-MM-DD" format.
        rng (np.random.Generator)   : NumPy random generator instance.

    Returns:
        pd.DataFrame: Monthly NMD dataset with columns:
            date, balance, deposit_rate, market_rate, repo_rate,
            unemployment, cds_spread, wealth, equity_return, equity_vol.
    """

    # Rates
    market_rate, repo_rate = generate_market_rates(n_months, rng)

    deposit_rate = generate_deposit_rate(
        market_rate = market_rate,
        beta = 0.30,
        lambda_up = 0.15,
        lambda_down = 0.45,
        spread = 0.005,
        rng = rng
    )

    # Macro variables
    unemployment = generate_unemployment(n_months, rng)
    cds_spread = generate_cds_spread(n_months, rng)

    # Balance
    balance = generate_balance(
        n = n_months,
        market_rate = market_rate,
        deposit_rate = deposit_rate,
        cds_spread = cds_spread,
        initial_balance = 5_000.0,
        trend_monthly = 0.01,
        rng = rng
    )

    # Wealth & equity 
    wealth, equity_return, equity_vol = generate_wealth_and_equity(n_months, rng)

    # Assemble DataFrame
    dates = pd.date_range(start = start_date, periods = n_months, freq = "ME")

    df = pd.DataFrame(
        {
            "date": dates,
            "balance": balance,
            "deposit_rate": deposit_rate,
            "market_rate": market_rate,
            "repo_rate": repo_rate,
            "unemployment": unemployment,
            "cds_spread": cds_spread,
            "wealth": wealth,
            "equity_return": equity_return,
            "equity_vol": equity_vol
        }
    ).set_index("date")

    return df



























































# Generate data for wealth allocation
def data_wealth() -> pd.DataFrame:
    
    """
    Generate wealth data.

    Description:
        To generate wealth data by assumed high risk level at equity.
        The bond is the most selected asset to be allocated.

    Args:
        None.

    Returns:
        pd.DataFrame: Generated wealth data for model.

    Notes:
        - N/A.
    """
    
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2014-01-01", periods = n, freq = "ME")

    # Asset returns / attractiveness
    r_deposit = 1.5 + np.random.normal(0, 0.1, n)
    r_bond = 3.0 + np.random.normal(0, 0.3, n)
    r_equity = 7.0 + np.random.normal(0, 2.0, n) #High-risk
    r_fund = 5.0 + np.random.normal(0, 1.0, n)

    sigma_equity = np.abs(np.random.normal(15, 3, n)) #Volatility

    # True shares (Dirichlet-distributed around logit model)
    V = np.stack(
        [
            0.5 + 0.8 * r_deposit - 0.3 * sigma_equity,
            0.2 + 0.5 * r_bond,
            -0.3 + 0.3 * r_equity - 0.8 * sigma_equity,
            0.1 + 0.4 * r_fund - 0.4 * sigma_equity,
        ],
        axis = 1
    )
    shares = softmax(V, axis = 1)
    shares += np.random.dirichlet([20, 20, 20, 20], n) * 0.05
    shares /= shares.sum(axis = 1, keepdims = True)

    # Total wealth (growing trend)
    total_wealth = np.linspace(1000, 1600, n) + np.random.normal(0, 20, n)

    # To DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "r_deposit": r_deposit,
            "r_bond": r_bond,
            "r_equity": r_equity,
            "r_fund": r_fund,
            "sigma_equity": sigma_equity,
            "share_deposit": shares[:, 0],
            "share_bond": shares[:, 1],
            "share_equity": shares[:, 2],
            "share_fund": shares[:, 3],
            "total_wealth": total_wealth
        }
    ).assign(deposit_balance = lambda d: d["total_wealth"] * d["share_deposit"])
    df.set_index("date", inplace = True)

    return df

# Generate data for ECM Model
def data_ecm() -> pd.DataFrame:

    """
    Generate for ECM Model.

    Description:
        To generate data for ECM Model. The data is contained the several rates cycle.
        For example, the policy rate, deposit rate, market rate.

    Args:
        None.

    Returns:
        pd.DataFrame: Generated data for ECM Model.

    Notes:
        - N/A.
    """

    np.random.seed(1)
    n = 120 #10 years monthly basis
    dates = pd.date_range("2014-01-01", periods = n, freq = "ME")

    # Parameters
    u1 = np.cumsum(np.random.normal(0, 0.05, n))
    u2 = np.cumsum(np.random.normal(0, 0.03, n))

    # Simulate realistic deposit balance + macro
    r_policy = 1.5 + u1 + np.random.normal(0, 0.02, n) #I(1)
    r_deposit = 0.4 * r_policy + np.cumsum(np.random.normal(0, 0.02, n))  #I(1), Pass-through 40%
    r_mmkt = 1.8 * r_policy + np.cumsum(np.random.normal(0, 0.02, n))  #I(1)
    gdp_growth = 0.03 + np.random.normal(0, 0.005, n) #I(0)
    ln_balance = 5.0 + 0.8 * r_deposit + u2 * 0.1 + np.random.normal(0, 0.03, n)

    # To DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "ln_balance": ln_balance,
            "r_deposit": r_deposit,
            "r_policy": r_policy,
            "r_mmkt": r_mmkt,
            "gdp_growth": gdp_growth
        }
    )
    df["balance"] = np.exp(df["ln_balance"])
    df.set_index("date", inplace = True)

    return df

# Generate Mock NMD Balance Data
def generate_nmd_data(
    n_accounts: int,
    n_months: int,
    seed: int
) -> pd.DataFrame:
    
    """
    Generate synthetic NMD balance panel data with real dates.

    Description:
        This function simulates deposit balances over time using
        month-end dates starting from 2014-01-31.

        Account IDs are formatted as A000001, A000002, etc.

    Args:
        n_accounts (int) : Number of accounts.
        n_months (int)   : Number of time periods.
        seed (int)       : Random seed.

    Returns:
        pd.DataFrame: Simulated panel data with datetime index.

    Notes:
        - Uses month-end frequency (M).
    """

    print("=== Processing ===\n[INFO]: Generating mock NMD data")

    np.random.seed(seed)
    data = []

    # Generate month-end date range
    dates = pd.date_range(start = "2014-01-31", periods = n_months, freq = "ME")

    for acc in range(1, n_accounts + 1):

        # Format account ID
        acc_id = f"A{acc:06d}"
        balance = np.random.uniform(800, 1600)

        for date in dates:
            # Cashflow
            inflow = np.random.uniform(0, 300)
            outflow = np.random.uniform(0, 300)
            
            # Balance
            balance = balance + inflow - outflow
            balance = max(balance, 0) #Non-negative
            data.append(
                {
                    "acc_id": acc_id,
                    "month": date,
                    "balance": balance
                }
            )

    df = pd.DataFrame(data)

    return df

# Generate rate model data
def generate_rate_data(
    n_months: int,
    seed: int
) -> pd.DataFrame:
    
    """
    Generate market rate and deposit rate data.

    Description:
        Simulates market rates with trend + noise and deposit rates with lagged response.

    Args:
        n_months (int)   : Number of time periods.
        seed (int)       : Random seed.

    Returns:
        pd.DataFrame
    """

    print("=== Processing ===\n[INFO]: Generate rate data")

    np.random.seed(seed)

    dates = pd.date_range(start = "2014-01-31", periods = n_months, freq = "ME")

    market = []
    deposit = []

    r_mkt = 0.02
    r_dep = 0.01

    for _ in range(n_months):

        # Market rate (random walk)
        shock = np.random.normal(0, 0.001)
        r_mkt = r_mkt + shock

        # Reposit responds slowly
        r_dep = 0.9 * r_dep + 0.1 * (0.5 * r_mkt)

        market.append(r_mkt)
        deposit.append(r_dep)

    df = pd.DataFrame(
        {
            "date": dates,
            "market_rate": market,
            "deposit_rate": deposit
        }
    )


    return df