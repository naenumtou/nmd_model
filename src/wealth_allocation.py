
import numpy as np
import pandas as pd
import warnings

from scipy.optimize import minimize

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Optimal wealth allocation
def optimise_allocation(
    core_balance: float,
    yields: dict,
    dur_cat: float,
    dur_float: float,
    dur_liq: float,
    rate_shock: float,
    eve_limit_pct: float,
    min_liq_pct: float,
    n_starts: int = 10
) -> dict:

    """
    Find optimal allocation across three strategies to maximise NII.

    Description:
        Solves:
            Maximise   NII = Σ(w_i x y_i) x core_balance
            Subject to Σw_i = 1
                       ΔEVE = Σ(w_i x dur_i) x core x shock ≤ eve_limit
                       w_liq ≥ min_liq_pct
                       w_i ≥ 0

        ΔEVE approximation uses linear duration sensitivity:
            ΔEVE ≈ −Σ(w_i x dur_i x core x shock)

        Uses multiple random starting points for global stability.

    Args:
        core_balance (float)    : Core balance in MB.
        yields (dict)           : Output from get_strategy_yields().
        dur_cat (float)         : Duration of caterpillar strategy (years).
        dur_float (float)       : Duration of floating strategy (years).
        dur_liq (float)         : Duration of liquidity buffer (years).
        rate_shock (float)      : Rate shock for ΔEVE calculation (decimal).
        eve_limit_pct (float)   : Max ΔEVE as % of core balance.
        min_liq_pct (float)     : Minimum liquidity buffer weight.
        n_starts (int)          : Number of random starting points (Default: 10).

    Returns:
        dict: {
            "weights"       : dict  — {cat, float, liq} optimal weights,
            "notional_mb"   : dict  — MB allocated per strategy,
            "nii_annual"    : float — annual NII (MB),
            "delta_eve"     : float — approximate ΔEVE (MB),
            "eve_limit"     : float — EVE limit used (MB),
            "nii_breakdown" : dict  — NII contribution per strategy,
            "converged"     : bool  — whether any starting point converged
        }
    """

    y = np.array([yields["cat"], yields["float"], yields["liq"]])
    dur = np.array([dur_cat, dur_float, dur_liq])
    eve_limit = eve_limit_pct * core_balance

    def neg_nii(w):
        return -float(np.dot(w, y) * core_balance)

    def delta_eve_fn(w):
        return float(np.dot(w, dur) * core_balance * rate_shock)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: eve_limit - delta_eve_fn(w)},
        {"type": "ineq", "fun": lambda w: w[2] - min_liq_pct}
    ]
    bounds = [(0.0, 1.0)] * 3

    best_nii = -np.inf
    best_w = None
    converged = False
    rng = np.random.default_rng(42)

    for _ in range(n_starts):
        w0 = rng.dirichlet([1, 1, 1])
        w0[2] = max(w0[2], min_liq_pct)
        w0 /= w0.sum()
        result = minimize(
            neg_nii,
            x0 = w0,
            method = "SLSQP",
            bounds = bounds, constraints = constraints,
            options = {"ftol": 1e-14, "maxiter": 3000}
        )
        nii = -result.fun
        if result.success and nii > best_nii:
            best_nii = nii
            best_w = result.x.copy()
            converged = True

    if best_w is None:
        best_w = np.array([1 - min_liq_pct, 0.0, min_liq_pct])

    w_cat, w_float, w_liq = best_w
    notional = {k: v*core_balance for k, v in zip(["cat","float","liq"], best_w)}

    return {
        "weights": {"cat": float(w_cat), "float": float(w_float), "liq": float(w_liq)},
        "notional_mb": notional,
        "nii_annual": float(best_nii),
        "delta_eve": float(delta_eve_fn(best_w)),
        "eve_limit": float(eve_limit),
        "nii_breakdown": {
            "cat": float(w_cat * yields["cat"] * core_balance),
            "float": float(w_float * yields["float"] * core_balance),
            "liq": float(w_liq * yields["liq"] * core_balance)
        },
        "converged": converged
    }

# Allocation different limit EVE
def allocation_sensitivity(
    core_balance: float,
    yields: dict,
    dur_cat: float,
    dur_float: float,
    dur_liq: float,
    rate_shock: float,
    min_liq_pct: float,
    eve_limit_range: list
) -> pd.DataFrame:

    """
    Show how optimal allocation changes across different EVE limit assumptions.

    Description:
        Runs optimise_allocation() for each eve_limit_pct in eve_limit_range
        and summarises the tradeoff between NII and ΔEVE risk.
        Useful for management to see the NII cost of tighter IRRBB limits.

    Args:
        core_balance (float)    : Core balance in MB.
        yields (dict)           : Output from get_strategy_yields().
        dur_cat (float)         : Caterpillar duration.
        dur_float (float)       : Floating duration.
        dur_liq (float)         : Liquidity buffer duration.
        rate_shock (float)      : Rate shock for ΔEVE.
        min_liq_pct (float)     : Minimum liquidity weight.
        eve_limit_range (list)  : List of EVE limit percentages to test.

    Returns:
        pd.DataFrame: Sensitivity table with allocation and NII per EVE limit.
    """

    rows = []
    for pct in eve_limit_range:
        res = optimise_allocation(
            core_balance = core_balance, yields = yields,
            dur_cat = dur_cat, dur_float = dur_float, dur_liq = dur_liq,
            rate_shock = rate_shock, eve_limit_pct = pct,
            min_liq_pct = min_liq_pct, n_starts = 5
        )
        rows.append(
            {
                "eve_limit_pct": f"{pct:.0%}",
                "eve_limit_mb": round(pct * core_balance, 1),
                "w_cat": round(res["weights"]["cat"] * 100, 2),
                "w_float": round(res["weights"]["float"] * 100, 2),
                "w_liq": round(res["weights"]["liq"] * 100, 2),
                "nii_annual_mb": round(res["nii_annual"], 2),
                "delta_eve_mb": round(res["delta_eve"], 2),
                "converged": res["converged"]
            }
        )
        
    return pd.DataFrame(rows).set_index("eve_limit_pct")

# LCR
def compute_lcr(
    core_balance: float,
    non_core_balance: float,
    alloc_result: dict,
    stable_pct_lcr: float = 0.75,
    outflow_rate_stable: float = 0.05,
    outflow_rate_unstable: float = 0.10,
    outflow_rate_noncash: float = 0.25,
    hqla_haircut: float = 1.0
) -> dict:

    """
    Compute Liquidity Coverage Ratio (LCR) from NMD allocation.

    Description:
        LCR = HQLA / Net Cash Outflow (30 days) >= 100%

        HQLA = liquidity buffer x hqla_haircut
            Assumes liquidity buffer is held as Level 1 HQLA (cash/govt bond).

        Net Cash Outflow = outflow from NMD deposits under 30-day stress:
            stable retail   : 5% of stable portion
            unstable retail : 10% of unstable portion
            non-core/wholesale: 25% of non-core balance

        Dynamic MIN_LIQ = total_outflow / core_balance
            Replaces the static 15% rule of thumb in optimise_allocation()
            with an LCR-derived minimum liquidity requirement.

    Args:
        core_balance (float)            : Core balance in MB.
        non_core_balance (float)        : Non-core balance in MB.
        alloc_result (dict)             : Output from optimise_allocation().
        stable_pct_lcr (float)          : % of core classified as stable (default: 75%).
        outflow_rate_stable (float)     : LCR outflow rate for stable retail (default: 5%).
        outflow_rate_unstable (float)   : LCR outflow rate for unstable retail (default: 10%).
        outflow_rate_noncash (float)    : LCR outflow rate for non-core (default: 25%).
        hqla_haircut (float)            : Haircut applied to liquidity buffer as HQLA (default: 1.0).

    Returns:
        dict: {
            "hqla"             : float — HQLA value (MB),
            "outflow_stable"   : float — 30-day outflow from stable deposits,
            "outflow_unstable" : float — 30-day outflow from unstable deposits,
            "outflow_noncash"  : float — 30-day outflow from non-core,
            "total_outflow"    : float — total net cash outflow,
            "lcr_pct"          : float — LCR in %,
            "min_liq_dynamic"  : float — LCR-derived minimum liquidity weight,
            "lcr_pass"         : bool  — True if LCR >= 100%
        }
    """

    hqla = alloc_result["notional_mb"]["liq"] * hqla_haircut

    outflow_stable   = core_balance * stable_pct_lcr * outflow_rate_stable
    outflow_unstable = core_balance * (1 - stable_pct_lcr) * outflow_rate_unstable
    outflow_noncash = non_core_balance * outflow_rate_noncash
    total_outflow = outflow_stable + outflow_unstable + outflow_noncash

    lcr_pct = (hqla / total_outflow * 100) if total_outflow > 0 else float("inf")
    min_liq_dynamic = total_outflow / core_balance

    return {
        "hqla": float(hqla),
        "outflow_stable": float(outflow_stable),
        "outflow_unstable": float(outflow_unstable),
        "outflow_noncash": float(outflow_noncash),
        "total_outflow": float(total_outflow),
        "lcr_pct": float(lcr_pct),
        "min_liq_dynamic": float(min_liq_dynamic),
        "lcr_pass": lcr_pct >= 100.0
    }

# NSFR
def compute_nsfr(
    core_balance: float,
    non_core_balance: float,
    alloc_result: dict,
    asf_factor_stable: float = 0.95,
    asf_factor_unstable: float = 0.90,
    asf_factor_noncash: float = 0.50,
    stable_pct_nsfr: float = 0.75,
    rsf_factor_cat: float = 0.85,
    rsf_factor_hqla: float = 0.05
) -> dict:

    """
    Compute Net Stable Funding Ratio (NSFR) from NMD Allocation.

    Description:
        NSFR = Available Stable Funding (ASF) / Required Stable Funding (RSF) >= 100%

        ASF from NMD deposits:
            stable retail NMD   : balance x 95%
            unstable retail NMD : balance x 90%
            non-core/wholesale  : balance x 50%

        RSF from assets:
            caterpillar (5Y bond) : notional x 85%  (long-term asset RSF)
            HQLA / liquidity buf  : notional x 5%   (Level 1 HQLA RSF)

        ASF and RSF factors follow BCBS liquidity standards.

    Args:
        core_balance (float)        : Core balance in MB.
        non_core_balance (float)    : Non-core balance in MB.
        alloc_result (dict)         : Output from optimise_allocation().
        asf_factor_stable (float)   : ASF factor for stable NMD (default: 95%).
        asf_factor_unstable (float) : ASF factor for unstable NMD (default: 90%).
        asf_factor_noncash (float)  : ASF factor for non-core (default: 50%).
        stable_pct_nsfr (float)     : % of core classified as stable (default: 75%).
        rsf_factor_cat (float)      : RSF factor for caterpillar bonds (default: 85%).
        rsf_factor_hqla (float)     : RSF factor for HQLA (default: 5%).

    Returns:
        dict: {
            "asf_stable"   : float — ASF from stable NMD,
            "asf_unstable" : float — ASF from unstable NMD,
            "asf_noncash"  : float — ASF from non-core,
            "asf_total"    : float — total ASF,
            "rsf_cat"      : float — RSF from caterpillar,
            "rsf_hqla"     : float — RSF from liquidity buffer,
            "rsf_total"    : float — total RSF,
            "nsfr_pct"     : float — NSFR in %,
            "nsfr_pass"    : bool  — True if NSFR >= 100%
        }
    """

    asf_stable   = core_balance * stable_pct_nsfr * asf_factor_stable
    asf_unstable = core_balance * (1 - stable_pct_nsfr) * asf_factor_unstable
    asf_noncash  = non_core_balance * asf_factor_noncash
    asf_total    = asf_stable + asf_unstable + asf_noncash

    rsf_cat  = alloc_result["notional_mb"]["cat"] * rsf_factor_cat
    rsf_hqla = alloc_result["notional_mb"]["liq"] * rsf_factor_hqla
    rsf_total = rsf_cat + rsf_hqla

    nsfr_pct = (asf_total / rsf_total * 100) if rsf_total > 0 else float("inf")

    return {
        "asf_stable": float(asf_stable),
        "asf_unstable": float(asf_unstable),
        "asf_noncash": float(asf_noncash),
        "asf_total": float(asf_total),
        "rsf_cat": float(rsf_cat),
        "rsf_hqla": float(rsf_hqla),
        "rsf_total": float(rsf_total),
        "nsfr_pct": float(nsfr_pct),
        "nsfr_pass": nsfr_pct >= 100.0
    }