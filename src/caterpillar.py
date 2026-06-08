
import numpy as np
import pandas as pd
import warnings

from scipy.stats import norm

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Caterpillar table
def build_caterpillar(
    core_balance: float,
    target_tenor_years: float,
    n_tranches: int,
    yield_curve: pd.DataFrame,
    wal_liability: float
) -> dict:

    """
    Build a caterpillar (rolling tranche) structural hedge schedule.

    Description:
        Divides core balance into n_tranches equal tranches. Each tranche
        is invested at the target tenor. Tranches are staggered by one year,
        so the portfolio has a constant average duration equal to:

            avg_duration = target_tenor - (n_tranches - 1) / 2

        This maintains duration alignment with the NMD liability WAL over time,
        unlike a one-off bond purchase where duration drifts down each year.

        The caterpillar yield is the weighted average of the yield at target
        tenor as each tranche was invested (here simplified to current yield).

        NII sensitivity is computed as:
            ΔNII(year 1) = tranche_size x rate_shock
        Only one tranche reprices per year (the one that matures and rolls).

    Args:
        core_balance (float)            : Core balance in MB.
        target_tenor_years (float)      : Investment tenor per tranche in years.
        n_tranches (int)                : Number of tranches (= roll frequency in years).
        yield_curve (pd.DataFrame)      : Output from build_yield_curve().
        wal_liability (float)           : WAL of core deposit liability from NB05.

    Returns:
        dict: {
            "tranche_size"         : float      — MB per tranche,
            "n_tranches"           : int        — number of tranches,
            "target_tenor"         : float      — investment tenor in years,
            "avg_duration"         : float      — average portfolio duration (years),
            "wal_liability"        : float      — liability WAL for comparison,
            "duration_gap"         : float      — avg_duration - wal_liability,
            "portfolio_yield"      : float      — current weighted yield,
            "nii_annual"           : float      — annual NII at current yield (MB),
            "schedule"             : pd.DataFrame — tranche schedule,
            "nii_sensitivity"      : dict       — ΔNII per rate shock scenario
        }
    """

    tranche_size = core_balance / n_tranches
    avg_duration = target_tenor_years - (n_tranches - 1) / 2
    duration_gap = avg_duration - wal_liability

    # Current yield at target tenor
    tenor_label = f"{int(target_tenor_years * 12)}M" if target_tenor_years < 1 else f"{int(target_tenor_years)}Y"
    portfolio_yield = float(yield_curve[tenor_label].iloc[-1])
    nii_annual = core_balance * portfolio_yield

    # Build schedule table
    schedule = pd.DataFrame(
        {
            "tranche": [f"T{i+1}" for i in range(n_tranches)],
            "invest_year": [i + 1 for i in range(n_tranches)],
            "maturity_year": [i + 1 + target_tenor_years for i in range(n_tranches)],
            "balance_mb": [round(tranche_size, 2)] * n_tranches,
            "yield_pct": [round(portfolio_yield * 100, 4)] * n_tranches,
            "duration_yr": [round(target_tenor_years - i, 2) for i in range(n_tranches)]    
        }
    )

    # NII Sensitivity: one tranche rolls per year -> reprices at new rate
    nii_sensitivity = {}
    for shock_bps in [100, 200, -100, -200]:
        shock = shock_bps / 10000
        delta_nii_year1 = tranche_size * shock #1 tranche reprices
        delta_nii_fullroll = core_balance * shock #all tranches rolled
        nii_sensitivity[f"+{shock_bps}bps" if shock_bps > 0 else f"{shock_bps}bps"] = {
            "delta_nii_year1_mb": round(delta_nii_year1, 4),
            "delta_nii_fullroll_mb": round(delta_nii_fullroll, 4),
            "pct_repricing_year1": round(1 / n_tranches, 4)
        }

    return {
        "tranche_size": tranche_size,
        "n_tranches": n_tranches,
        "target_tenor": target_tenor_years,
        "avg_duration": avg_duration,
        "wal_liability": wal_liability,
        "duration_gap": duration_gap,
        "portfolio_yield": portfolio_yield,
        "nii_annual": nii_annual,
        "schedule": schedule,
        "nii_sensitivity": nii_sensitivity
    }

# Maximises portfolio yield
def optimise_caterpillar(
    core_balance: float,
    wal_liability: float,
    yield_curve: pd.DataFrame,
    tenor_options: list,
    n_tranche_options: list,
    max_duration_gap: float = 0.5
) -> dict:

    """
    Find the caterpillar configuration that maximises portfolio yield
    subject to duration gap constraint.

    Description:
        Searches over all combinations of tenor and n_tranches to find
        the one that:

            Maximise  portfolio_yield
            Subject to |avg_duration - wal_liability| <= max_duration_gap

        Longer tenors earn more yield but may cause duration mismatch.
        More tranches reduce the mismatch but also reduce avg_duration.

    Args:
        core_balance (float)        : Core balance in MB.
        wal_liability (float)       : Liability WAL from NB05 (years).
        yield_curve (pd.DataFrame)  : Output from build_yield_curve().
        tenor_options (list)        : Candidate tenors in years.
        n_tranche_options (list)    : Candidate number of tranches.
        max_duration_gap (float)    : Max |asset_dur - liability_wal| (default: 0.5Y).

    Returns:
        dict: {
            "best"      : dict         — best caterpillar result,
            "grid"      : pd.DataFrame — full grid of all candidates,
            "feasible"  : pd.DataFrame — feasible candidates only
        }
    """

    rows = []
    for tenor in tenor_options:
        for n_t in n_tranche_options:
            result = build_caterpillar(
                core_balance = core_balance,
                target_tenor_years = tenor,
                n_tranches = n_t,
                yield_curve = yield_curve,
                wal_liability = wal_liability
            )
            rows.append(
                {
                    "tenor": tenor,
                    "n_tranches": n_t,
                    "avg_duration": round(result["avg_duration"], 3),
                    "duration_gap": round(result["duration_gap"], 3),
                    "yield_pct": round(result["portfolio_yield"] * 100, 4),
                    "nii_annual_mb": round(result["nii_annual"], 2),
                    "feasible": (abs(result["duration_gap"]) <= max_duration_gap) & (result["avg_duration"] >= wal_liability)
                }
            )

    grid = pd.DataFrame(rows)
    feasible = grid[grid["feasible"]].copy()

    if feasible.empty:
        best_row = grid.loc[grid["yield_pct"].idxmax()]
    else:
        best_row = feasible.loc[feasible["yield_pct"].idxmax()]

    best = build_caterpillar(
        core_balance = core_balance,
        target_tenor_years = float(best_row["tenor"]),
        n_tranches = int(best_row["n_tranches"]),
        yield_curve = yield_curve,
        wal_liability = wal_liability
    )

    return {"best": best, "grid": grid, "feasible": feasible}