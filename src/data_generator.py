
import numpy as np
import pandas as pd
import warnings

from scipy.special import softmax

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

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

    np.random.seed(1001)
    n = 120 #10 years monthly basis
    dates = pd.date_range("2014-01-01", periods = n, freq = "ME")

    # Simulate realistic deposit balance + macro
    r_policy = np.concatenate(
        [
            np.linspace(2.5, 1.5, 30), #Easing
            np.linspace(1.5, 0.5, 30), #Low rate era
            np.linspace(0.5, 4.0, 30), #Hiking cycle
            np.linspace(4.0, 3.0, 30), #Normalizing
        ]
    )

    r_deposit = r_policy * 0.4 + np.random.normal(0, 0.05, n)  #Pass-through ~40%
    r_mmkt = r_policy * 0.9 + np.random.normal(0, 0.05, n)
    r_gap = r_deposit - r_mmkt
    gdp_growth = 0.03 + np.random.normal(0, 0.005, n)
    cpi = 0.02 + np.random.normal(0, 0.003, n)

    # Balance: trend + rate response + seasonality + noise
    trend = np.linspace(100, 160, n)
    rate_effect = 5 * r_gap #Negative gap → outflow
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 12)
    balance = trend + rate_effect + seasonal + np.random.normal(0, 1.5, n)
    balance = np.maximum(balance, 50)

    # To DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "balance": balance,
            "r_deposit": r_deposit,
            "r_policy": r_policy,
            "r_mmkt": r_mmkt,
            "r_gap": r_gap,
            "gdp_growth": gdp_growth,
            "cpi": cpi
        }
    )
    df["ln_balance"] = np.log(df["balance"])
    df.set_index("date", inplace = True)
    
    return df
