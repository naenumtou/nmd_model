
import numpy as np
import pandas as pd
import warnings

from scipy.stats import norm

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Bachelier Model
def bachelier_floorlet(
    F: float,
    K: float,
    T: float,
    sigma_n: float,
    P: float,
    notional: float
) -> float:

    """
    Price a single floorlet using the Bachelier (normal) model.

    Description:
        Bachelier model assumes rate follows arithmetic (normal) Brownian motion.
        Preferred over Black's log-normal model for near-zero or negative rates,
        and for NMD where strike is often far below the current forward rate.

        Formula:
            Floorlet = notional x P x [(K-F)·N(-d) + rho_n·√T·n(d)]
            d = (F - K) / (rho_n · √T)

        When T → 0: Floorlet → notional x P x max(K-F, 0)  (intrinsic value)

    Args:
        F (float)           : Forward rate for the period.
        K (float)           : Floor strike rate.
        T (float)           : Time to expiry in years.
        sigma_n (float)     : Annualised normal (absolute) volatility.
        P (float)           : Discount factor P(0, T).
        notional (float)    : Notional balance (MB).

    Returns:
        float: Floorlet value in MB.
    """

    if T <= 0:
        return float(notional * P * max(K - F, 0))

    vol_sqrt_t = sigma_n * np.sqrt(T)
    if vol_sqrt_t < 1e-10:
        return float(notional * P * max(K - F, 0))

    d = (F - K) / vol_sqrt_t
    value = notional * P * ((K - F) * norm.cdf(-d) + vol_sqrt_t * norm.pdf(d))
    
    return float(value)

# Price the NMD floor
def price_nmd_floor(
    balance_profile: np.ndarray,
    forward_rates: np.ndarray,
    sigma_n: float,
    r0: float,
    strike: float = 0.0,
    n_months: int = 60
) -> dict:

    """
    Price the NMD floor as a sum of Bachelier floorlets.

    Description:
        Iterates over each monthly period and prices one floorlet per period:

            Floor = Σ_{t=1}^{n_months} Floorlet(F_t, K, t/12, rho_n, P_t, D_t)

        Discount factors use a flat curve at r0:
            P_t = exp(-r0 x t/12)

        The floor value represents the implicit benefit the bank receives from
        depositors who accept a deposit rate that cannot go below the strike —
        equivalent to the cost of purchasing this floor in the derivatives market.

    Args:
        balance_profile (np.ndarray)    : Core balance at start of each month (MB).
        forward_rates (np.ndarray)      : Forward rate per period.
        sigma_n (float)                 : Annualised normal vol.
        r0 (float)                      : Current rate for discount curve.
        strike (float)                  : Floor strike (default: 0.0 = zero-floor).
        n_months (int)                  : Number of periods (default: 60).

    Returns:
        dict: {
            "floor_value"       : float      — total floor value (MB),
            "floorlet_values"   : np.ndarray — value per period (MB),
            "discount_factors"  : np.ndarray — P(0,t) per period,
            "pct_of_core"       : float      — floor value as % of core balance,
            "strike"            : float      — strike used
        }
    """

    months = np.arange(1, n_months + 1)
    discount_factors = np.exp(-r0 * months / 12)

    floorlet_values = np.array([
        bachelier_floorlet(
            F = forward_rates[t],
            K = strike,
            T = (t + 1) / 12,
            sigma_n = sigma_n,
            P = discount_factors[t],
            notional = balance_profile[t]
        )
        for t in range(n_months)
    ])

    floor_value = float(floorlet_values.sum())
    core_balance = float(balance_profile[0])

    return {
        "floor_value": floor_value,
        "floorlet_values": floorlet_values,
        "discount_factors": discount_factors,
        "pct_of_core": floor_value / core_balance if core_balance > 0 else 0.0,
        "strike": strike
    }

# Greeks of the NMD floor
def floor_sensitivity(
    balance_profile: np.ndarray,
    forward_rates: np.ndarray,
    sigma_n: float,
    r0: float,
    strike: float = 0.0,
    bump_r: float = 0.0001,
    bump_sigma: float = 0.0001
) -> dict:

    """
    Compute Greeks of the NMD floor via finite difference bumps.

    Description:
        Delta  : dFloor/dr0         — sensitivity to parallel rate shift (+1bps)
        Vega   : dFloor/d(sigma)    — sensitivity to vol shift (+1bps normal vol)

        Both are computed via central difference:
            Greek = (Floor(x + bump) - Floor(x - bump)) / (2 x bump)

    Args:
        balance_profile (np.ndarray)    : Core balance per month (MB).
        forward_rates (np.ndarray)      : Forward rates per period.
        sigma_n (float)                 : Annualised normal vol.
        r0 (float)                      : Current rate.
        strike (float)                  : Floor strike (default: 0.0).
        bump_r (float)                  : Rate bump for delta (default: 1bps).
        bump_sigma (float)              : Vol bump for vega (default: 1bps).

    Returns:
        dict: {
            "delta"     : float — dFloor/dr0 per 1bps rate shift (MB per bps),
            "vega"      : float — dFloor/d(sigma) per 1bps vol shift (MB per bps),
            "delta_pct" : float — delta as % of floor value,
            "vega_pct"  : float — vega as % of floor value
        }
    """

    def floor_val(fwd, sig, r):
        return price_nmd_floor(balance_profile, fwd, sig, r, strike)["floor_value"]

    base = floor_val(forward_rates, sigma_n, r0)

    # Delta: bump forward rates and discount rate together
    fwd_up   = forward_rates + bump_r
    fwd_dn   = forward_rates - bump_r
    delta = (floor_val(fwd_up, sigma_n, r0 + bump_r) -
             floor_val(fwd_dn, sigma_n, r0 - bump_r)) / (2 * bump_r) * 0.0001

    # Vega: bump sigma
    vega = (floor_val(forward_rates, sigma_n + bump_sigma, r0) -
            floor_val(forward_rates, sigma_n - bump_sigma, r0)) / (2 * bump_sigma) * 0.0001

    return {
        "delta_per_bps": float(delta),
        "vega_per_bps": float(vega),
        "delta_pct": float(delta / base) if base > 0 else 0.0,
        "vega_pct": float(vega / base) if base > 0 else 0.0
    }