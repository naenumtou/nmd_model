
import numpy as np
import pandas as pd
import warnings

from scipy.special import softmax


warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# MNL Estimation
def mnl_loss(
    params: np.ndarray,
    X: np.ndarray,
    shares_obs: np.ndarray
) -> float:
    
    """
    MNL Estimation loss function.

    Description:
        Loss function for optimization.

    Args:
        params (np.ndarray)     : Parameters of MNL Model.
        X (np.ndarray)          : Features for modeling. The returns on asset classes.
        shares_obs (np.ndarray) : Allocation percent on asset classes.

    Returns:
        float: Average loss in the model.

    Notes:
        - N/A.
    """
    
    K = shares_obs.shape[1]
    params_mat = params.reshape(K, X.shape[1])
    shares_pred = softmax(X @ params_mat.T, axis = 1)

    return -np.mean(
        np.sum(
            shares_obs * np.log(shares_pred + 1e-10),
            axis = 1
        )
    )

# Projection wealth balance
def project_balance_wealth(
    df_hist: pd.DataFrame,
    params_est: np.ndarray,
    scenario_rates: dict,
    horizon: int = 12
) -> pd.DataFrame:
    
    """
    Projection wealth balance.

    Description:
        Projection wealth balance from latest information.

    Args:
        df_hist (pd.DataFrame)  : Historical DataFrame (Using latest "total_wealth").
        params_est (np.ndarray) : (K, p) estimated MNL params.
        scenario_rates (dict)   : Dict of arrays length horizon.
        horizon (int)           : Projection horizon. Default is 12 months. (1-Year ahead).

    Returns:
        pd.DataFrame: Projected wealth balance.

    Notes:
        - N/A.
    """

    last_wealth = df_hist["total_wealth"].iloc[-1]
    wealth_path = np.linspace(last_wealth, last_wealth * 1.05, horizon)
    X_scen = np.column_stack(
        [
                np.ones(horizon),
                scenario_rates["r_deposit"],
                scenario_rates["r_bond"],
                scenario_rates["r_equity"],
                scenario_rates["r_fund"],
                scenario_rates["sigma_equity"]
        ]
    )
    shares_scen = softmax(X_scen @ params_est.T, axis = 1)
    future_dates = pd.date_range(
        df_hist.index[-1] + pd.offsets.MonthEnd(1),
        periods = horizon,
        freq = "ME"
    )

    return pd.DataFrame(
        {
            "total_wealth": wealth_path,
            "share_deposit": shares_scen[:, 0],
            "deposit_balance": wealth_path * shares_scen[:, 0],
            **{f"share_{a}": shares_scen[:, i]
            for i, a in enumerate(["deposit", "bond", "equity", "fund"])}
        },
        index = future_dates
    )