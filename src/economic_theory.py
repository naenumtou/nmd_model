
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Carlibration
def calibrate_all(
    market_rate: pd.Series,
    deposit_rate: pd.Series,
    balance: pd.Series,
    cds_spread: pd.Series,
    beta_2: float,
    gamma: float,
    alpha: float
) -> dict:

    """
    Calibrate all four driver equations from historical data.

    Description:
        Estimates parameters for each dynamic equation and builds the Cholesky
        factor from residual correlations to preserve co-movement in simulation.

        Equations:
        1. Rate AR(1):    Δr_t = μ·Δr_{t-1} + ε_r
        2. Deposit ECM:   Δd_t = β·Δr_t + κ·(d_{t-1} - gamma·r_{t-1} - rho) + ε_d
        3. Volume:        Δln(D_t) = φ + λ·(r_t - d_t) + θ·1(CDS>50bps) + ε_v
        4. CDS log AR(1): Δln(CDS_t) = η - rho·ln(CDS_{t-1}) + ε_c

    Args:
        market_rate (pd.Series)     : Market rate series.
        deposit_rate (pd.Series)    : Client deposit rate series.
        balance (pd.Series)         : Deposit balance series.
        cds_spread (pd.Series)      : CDS spread series (bps).
        beta_2 (float)              : Short-term pass-through from JVD Model.
        gamma (float)               : Long-run pass-through from Threshold Model.
        alpha (float)               : Bank spread from Beta Regression.

    Returns:
        dict: {
            "rate"    : dict — {mu, sigma_r, r0},
            "deposit" : dict — {kappa, sigma_d, d0},
            "volume"  : dict — {phi, lam, theta, sigma_v, D0},
            "cds"     : dict — {eta, rho, sigma_c, cds0},
            "chol"    : np.ndarray — lower Cholesky factor (4x4)
        }
    """

    r = market_rate.values
    d = deposit_rate.values
    D = balance.values
    cds = cds_spread.values

    # Rate AR(1) on changes
    dr = np.diff(r)
    mu = float(np.linalg.lstsq(dr[:-1].reshape(-1, 1), dr[1:], rcond = None)[0][0])
    eps_r = dr[1:] - mu * dr[:-1]

    # Deposit kappa via error correction
    dd = np.diff(d)
    lhs = dd - beta_2 * dr[:len(dd)]
    ect = d[:-1] - gamma * r[:-1] - alpha
    kappa = float(np.linalg.lstsq(ect.reshape(-1, 1), lhs, rcond = None)[0][0])
    eps_d = lhs - kappa * ect

    # Volume OLS - Δ(r-d) avoids spurious level correlation
    dlogD = np.diff(np.log(D))
    dspread = np.diff(r - d)
    cds_dummy = (cds[:len(dlogD)] > 50).astype(float)
    Xv = np.column_stack([np.ones(len(dlogD)), dspread, cds_dummy])
    bv = np.linalg.lstsq(Xv, dlogD, rcond = None)[0]
    eps_v = dlogD - Xv @ bv

    # CDS log AR(1)
    lc = np.log(cds)
    dlc = np.diff(lc)
    Xc = np.column_stack([np.ones(len(dlc)), lc[:-1]])
    bc = np.linalg.lstsq(Xc, dlc, rcond = None)[0]
    eps_c = dlc - Xc @ bc

    # Cholesky from residuals
    min_len = min(len(eps_r), len(eps_d[1:]), len(eps_v[1:]), len(eps_c[1:]))
    R = np.column_stack(
        [
            eps_r[-min_len:],
            eps_d[1:][-min_len:],
            eps_v[1:][-min_len:],
            eps_c[1:][-min_len:]
        ]
    )
    corr_mat = np.corrcoef(R.T)
    corr_mat = (corr_mat + corr_mat.T) / 2
    np.fill_diagonal(corr_mat, 1.0)
    chol = np.linalg.cholesky(corr_mat)

    return {
        "rate": {"mu": mu, "sigma_r": float(np.std(eps_r)), "r0": float(r[-1])},
        "deposit": {"kappa": kappa, "sigma_d": float(np.std(eps_d)), "d0": float(d[-1])},
        "volume": {"phi": float(bv[0]), "lam": float(bv[1]), "theta": float(bv[2]),
                   "sigma_v": float(np.std(eps_v)), "D0": float(D[-1])},
        "cds": {"eta": float(bc[0]), "rho": float(-bc[1]),
                "sigma_c": float(np.std(eps_c)), "cds0": float(cds[-1])},
        "chol": chol
    }

# Simulation
def simulate_paths(
    params: dict,
    beta_2: float,
    gamma: float,
    alpha: float,
    n_paths: int = 1000,
    n_months: int = 60,
    rate_shock: float = 0.0,
    seed: int = 42
) -> dict:

    """
    Simulate Monte Carlo paths for all four NMD drivers.

    Description:
        Generates n_paths x n_months simulations using the calibrated equations:
        1. r_t  = max(r_{t-1} + μ·Δr_{t-1} + ε_r,  -0.006)
        2. d_t  = max(d_{t-1} + β·Δr_t + κ·ect_{t-1} + ε_d,  0.001)
        3. D_t  = D_{t-1} · exp(φ + λ·Δ(r_t-d_t) + θ·1(cds>50) + ε_v)
        4. CDS_t = via log AR(1)

        Volume uses Δ(r-d) consistent with calibration.
        Correlated shocks: ε = chol @ (rho ⊙ z),  z ~ N(0, I_4)

    Args:
        params (dict): Output from calibrate_all().
        beta_2 (float): Short-term pass-through.
        gamma (float): Long-run pass-through.
        alpha (float): Bank spread.
        n_paths (int): Number of simulation paths (default: 1000).
        n_months (int): Horizon in months (default: 60).
        rate_shock (float): Parallel shift to starting rate (default: 0).
        seed (int): Random seed (default: 42).

    Returns:
        dict: {
            "r"             : np.ndarray — rate paths (n_paths, n_months),
            "d"             : np.ndarray — deposit rate paths,
            "D"             : np.ndarray — balance paths,
            "cds"           : np.ndarray — CDS spread paths,
            "rate_shock"    : float      — shock applied
        }
    """

    rng = np.random.default_rng(seed)
    rp_ = params["rate"]
    dp_ = params["deposit"]
    vp_ = params["volume"]
    cp_ = params["cds"]
    chol = params["chol"]
    sigma = np.array([rp_["sigma_r"], dp_["sigma_d"], vp_["sigma_v"], cp_["sigma_c"]])

    r_paths = np.empty((n_paths, n_months))
    d_paths = np.empty((n_paths, n_months))
    D_paths = np.empty((n_paths, n_months))
    c_paths = np.empty((n_paths, n_months))

    for p in range(n_paths):
        z = rng.standard_normal((4, n_months))
        eps = (chol @ (sigma[:, None] * z)).T

        r_p = np.empty(n_months)
        d_p = np.empty(n_months)
        D_p = np.empty(n_months)
        c_p = np.empty(n_months)

        r_p[0] = rp_["r0"] + rate_shock
        d_p[0] = dp_["d0"]
        D_p[0] = vp_["D0"]
        c_p[0] = cp_["cds0"]
        dr_prev = 0.0
        prev_spread = r_p[0] - d_p[0]

        for t in range(1, n_months):
            dr_t = rp_["mu"] * dr_prev + eps[t, 0]
            r_p[t] = max(r_p[t - 1] + dr_t, -0.006)
            dr_prev = dr_t

            ect_t = d_p[t - 1] - gamma * r_p[t - 1] - alpha
            d_p[t] = max(d_p[t - 1] + beta_2 * dr_t + dp_["kappa"] * ect_t + eps[t, 1], 0.001)

            curr_spread = r_p[t] - d_p[t]
            dspread_t = curr_spread - prev_spread
            prev_spread = curr_spread
            cds_dd = 1.0 if c_p[t - 1] > 50 else 0.0
            D_p[t] = D_p[t - 1] * np.exp(
                vp_["phi"] + vp_["lam"] * dspread_t + vp_["theta"] * cds_dd + eps[t, 2]
            )

            lc_prev = np.log(max(c_p[t - 1], 1.0))
            c_p[t] = np.clip(
                np.exp(lc_prev + cp_["eta"] + cp_["rho"] * lc_prev + eps[t, 3]),
                1.0, 2000.0
            )

        r_paths[p] = r_p
        d_paths[p] = d_p
        D_paths[p] = D_p
        c_paths[p] = c_p

    return {
        "r": r_paths, "d": d_paths,
        "D": D_paths, "cds": c_paths,
        "rate_shock": rate_shock
    }

# EVE
def compute_eve(
    paths: dict,
    discount_method: str = "fixed"
) -> dict:

    """
    Compute Economic Value of NMD benefits from simulated paths.

    Description:
        Cash flow uses spread-minus-shock to isolate NMD benefit:
            CF_t = D_t x ((r_t - shock) - d_t)

        Stripping shock from r ensures CF measures sensitivity of NMD value
        to rate change, not windfall from higher rate level. This is consistent
        with IRRBB EVE sensitivity concept under BCBS 368.

        Two discount methods are supported:
        - fixed : flat curve at starting rate of the scenario (IRRBB standard)
                  discount_t = (1 + r0/12)^t
        - path  : path-specific cumulative product
                  discount_t = prod(1 + r_s/12) for s=1..t

    Args:
        paths (dict)            : Output from simulate_paths().
        discount_method (str)   : "fixed" or "path" (default: "fixed").

    Returns:
        dict: {
            "eve"      : float      — mean PV of benefits,
            "pv_paths" : np.ndarray — PV per path,
            "cf_mean"  : np.ndarray — mean monthly cash flow,
            "pv_p5"    : float      — 5th percentile PV,
            "pv_p95"   : float      — 95th percentile PV
        }
    """

    r = paths["r"]
    d = paths["d"]
    D = paths["D"]
    shock = paths["rate_shock"]

    # CF: Strip shock from r to measure NMD sensitivity not rate level
    cf = D * ((r - shock) - d)

    # Discount
    if discount_method == "fixed":
        r0 = float(r[:, 0].mean()) #Starting rate of scenario
        months = np.arange(1, r.shape[1] + 1)
        discount = (1 + r0 / 12) ** months #Shape (n_months,)
        pv_paths = np.sum(cf / discount[None, :], axis = 1)
    else:
        discount = np.cumprod(1 + r / 12, axis = 1)
        pv_paths = np.sum(cf / discount, axis = 1)

    return {
        "eve": float(np.mean(pv_paths)),
        "pv_paths": pv_paths,
        "cf_mean": cf.mean(axis = 0),
        "pv_p5": float(np.percentile(pv_paths, 5)),
        "pv_p95": float(np.percentile(pv_paths, 95))
    }