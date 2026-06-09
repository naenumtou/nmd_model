
import pandas as pd
import warnings

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# IRRBB Report
def build_irrbb_report(
    nb05: dict,
    nb06: dict,
    nb07: dict,
    nb08_09: dict,
    rate_shocks: list,
    stable_pct: float,
    beta_2: float
) -> dict:

    """
    Assemble IRRBB report tables from all upstream outputs.

    Description:
        Combines outputs from NB05-NB09 into standardised IRRBB tables:
        1. EVE sensitivity table
        2. NII sensitivity table
        3. Repricing gap table
        4. NMD model summary table

    Args:
        nb05 (dict)         : Output from reproduce_nb05().
        nb06 (dict)         : Output from reproduce_nb06().
        nb07 (dict)         : Output from reproduce_nb07().
        nb08_09 (dict)      : Output from reproduce_nb08_nb09().
        rate_shocks (list)  : List of rate shocks used in nb06.
        stable_pct (float)  : Stable fraction.
        beta_2 (float)      : Pass-through.

    Returns:
        dict: {
            "eve_table"      : pd.DataFrame,
            "nii_table"      : pd.DataFrame,
            "repricing_gap"  : pd.DataFrame,
            "nmd_summary"    : pd.DataFrame
        }
    """

    core = nb05["core_balance"]
    non_core = nb05["non_core_balance"]
    tranche_size = nb08_09["tranche_size"]
    w_cat = nb08_09["w_cat"]

    # 1. EVE table
    eve_rows = []
    for shock in rate_shocks:
        eve = nb06["eve_per_shock"][shock]
        delta = nb06["delta_eve"].get(shock, 0.0)
        eve_rows.append(
            {
                "scenario": f"+{int(shock*10000)}bps" if shock > 0 else (f"{int(shock*10000)}bps" if shock < 0 else "Base"),
                "shock": shock,
                "EVE (MB)": round(eve, 1),
                "ΔEVE (MB)": round(delta, 1),
                "ΔEVE (%)": f"{delta / abs(nb06['eve_base'])*100:.2f}%" if shock != 0.0 else "—"
            }
        )
    eve_table = pd.DataFrame(eve_rows).set_index("scenario")

    # 2. NII sensitivity table
    nii_rows = []
    for label, shock_bps in [("+100bps",100),("+200bps",200),("-100bps",-100),("-200bps",-200)]:
        shock_ = shock_bps / 10000
        delta_y1 = tranche_size * shock_ * w_cat
        delta_full = core * shock_ * w_cat
        nii_rows.append(
            {
                "scenario": label,
                "NII base (MB)": round(nb08_09["nii_alloc"], 1),
                "ΔNII year 1 (MB)": round(delta_y1, 2),
                "ΔNII fully rolled (MB)": round(delta_full, 2),
                "% repricing/year": f"{1/5:.2%}"
            }
        )
    nii_table = pd.DataFrame(nii_rows).set_index("scenario")

    # 3. Repricing gap
    buckets = nb05["irrbb_buckets"]
    gap_rows = []
    gap_rows.append(
        {
            "bucket": "Overnight (Non-core)", "balance_mb": round(non_core, 1),
            "pct_of_total": f"{non_core / (core + non_core)*100:.2f}%","description": "Reprice immediately"
        }
    )
    
    for label, val in buckets.items():
        gap_rows.append(
            {
                "bucket": label, "balance_mb": round(val, 1),
                "pct_of_total": f"{val/(core+non_core)*100:.2f}%","description": "Core — behavioral maturity"
            }
        )
    repricing_gap = pd.DataFrame(gap_rows).set_index("bucket")

    # 4. NMD model summary
    summary_data = {
        "Total balance (MB)": round(core + non_core, 1),
        "Stable % (NB03)": f"{stable_pct:.2%}",
        "Beta / Pass-through (NB04)": f"{beta_2:.4f}",
        "Core balance (MB)": round(core, 1),
        "Non-core balance (MB)": round(non_core, 1),
        "WAL — liability (years)": round(nb05["wal_years"], 3),
        "Base EVE (MB)": round(nb06["eve_base"], 1),
        "ΔEVE +200bps (MB)": round(nb06["delta_eve"].get(0.02, 0), 1),
        "Floor value K=0% (MB)": round(nb07["floor_zero"], 2),
        "Floor value K=d0 (MB)": round(nb07["floor_atm"], 2),
        "Caterpillar tenor": "5Y / 5 tranches",
        "Caterpillar avg duration": f"{nb08_09['cat_duration']:.2f}Y",
        "Caterpillar yield": f"{nb08_09['cat_yield']:.2%}",
        "Allocation w_cat": f"{nb08_09['w_cat']:.2%}",
        "Allocation w_liq": f"{nb08_09['w_liq']:.2%}",
        "NII annual (alloc, MB)": round(nb08_09["nii_alloc"], 1),
    }
    nmd_summary = pd.DataFrame.from_dict(summary_data, orient = "index", columns = ["Value"])

    return {
        "eve_table": eve_table,
        "nii_table": nii_table,
        "repricing_gap": repricing_gap,
        "nmd_summary": nmd_summary
    }

# Net repricing gap report
def build_net_repricing_gap(
    liability_buckets: dict,
    asset_repricing: dict
) -> pd.DataFrame:

    """
    Compute net repricing gap as asset minus liability per IRRBB time bucket.

    Description:
        Net gap = asset repricing - liability repricing per bucket.

        Positive net gap (asset heavy): asset reprices more than liability
            → rate rise benefits NII in that bucket
        Negative net gap (liability heavy): liability reprices more
            → rate rise hurts NII in that bucket

        Cumulative gap shows overall interest rate position:
            Positive cumulative → asset-sensitive bank (benefits from rate rise)
            Negative cumulative → liability-sensitive bank (benefits from rate fall)

    Args:
        liability_buckets (dict)    : IRRBB liability buckets from reproduce_nb05().
        asset_repricing (dict)      : Asset repricing profile from reproduce_nb08_nb09().

    Returns:
        pd.DataFrame: Net repricing gap table with columns
            [asset_mb, liability_mb, net_gap_mb, cumulative_gap_mb, position]
    """

    buckets = list(liability_buckets.keys())
    rows = []
    cumulative = 0.0
    for bucket in buckets:
        asset = asset_repricing.get(bucket, 0.0)
        liab = liability_buckets[bucket]
        net = asset - liab
        cumulative += net
        rows.append(
            {
                "bucket": bucket,
                "asset_mb": round(asset, 1),
                "liability_mb": round(liab, 1),
                "net_gap_mb": round(net, 1),
                "cumulative_mb": round(cumulative, 1),
                "position": "Asset heavy" if net > 0 else "Liability heavy" if net < 0 else "Balanced"
            }
        )
    return pd.DataFrame(rows).set_index("bucket")

# LCR and NFSR Report
def compute_liquidity_ratios(
    core_balance: float,
    non_core_balance: float,
    nb08_09: dict,
    stable_pct_lcr: float = 0.75,
    outflow_rate_stable: float = 0.05,
    outflow_rate_unstable: float = 0.10,
    outflow_rate_noncash: float = 0.25,
    asf_factor_stable: float = 0.95,
    asf_factor_unstable: float = 0.90,
    asf_factor_noncash: float = 0.50,
    rsf_factor_cat: float = 0.85,
    rsf_factor_hqla: float = 0.05
) -> dict:

    """
    Compute LCR and NSFR from NMD allocation and balance sheet.

    Description:
        LCR = HQLA / Net Cash Outflow (30 days) >= 100%
            HQLA = liquidity buffer allocation
            Outflow rates follow BCBS liquidity standards.

        NSFR = Available Stable Funding / Required Stable Funding >= 100%
            ASF from NMD deposits (stable 95%, unstable 90%, non-core 50%)
            RSF from asset investments (5Y bond 85%, HQLA 5%)

        Dynamic MIN_LIQ = total_outflow / core_balance
            Shows whether the static 15% rule is sufficient for LCR compliance.

    Args:
        core_balance (float)            : Core balance in MB.
        non_core_balance (float)        : Non-core balance in MB.
        nb08_09 (dict)                  : Output from reproduce_nb08_nb09().
        stable_pct_lcr (float)          : % of core classified stable for LCR (default: 75%).
        outflow_rate_stable (float)     : LCR outflow rate stable retail (default: 5%).
        outflow_rate_unstable (float)   : LCR outflow rate unstable retail (default: 10%).
        outflow_rate_noncash (float)    : LCR outflow rate non-core (default: 25%).
        asf_factor_stable (float)       : NSFR ASF factor stable NMD (default: 95%).
        asf_factor_unstable (float)     : NSFR ASF factor unstable NMD (default: 90%).
        asf_factor_noncash (float)      : NSFR ASF factor non-core (default: 50%).
        rsf_factor_cat (float)          : NSFR RSF factor 5Y bond (default: 85%).
        rsf_factor_hqla (float)         : NSFR RSF factor HQLA (default: 5%).

    Returns:
        dict: {
            "lcr"  : dict — LCR components and ratio,
            "nsfr" : dict — NSFR components and ratio,
            "summary" : pd.DataFrame — combined summary table
        }
    """

    hqla = nb08_09["w_liq"] * core_balance
    outflow_stable = core_balance * stable_pct_lcr * outflow_rate_stable
    outflow_unstable = core_balance * (1 - stable_pct_lcr) * outflow_rate_unstable
    outflow_noncash = non_core_balance * outflow_rate_noncash
    total_outflow = outflow_stable + outflow_unstable + outflow_noncash
    lcr_pct = hqla / total_outflow * 100 if total_outflow > 0 else float("inf")
    min_liq_dynamic = total_outflow / core_balance

    asf_stable = core_balance * stable_pct_lcr * asf_factor_stable
    asf_unstable = core_balance * (1 - stable_pct_lcr) * asf_factor_unstable
    asf_noncash = non_core_balance * asf_factor_noncash
    asf_total = asf_stable + asf_unstable + asf_noncash
    rsf_cat = nb08_09["w_cat"] * core_balance * rsf_factor_cat
    rsf_hqla = nb08_09["w_liq"] * core_balance * rsf_factor_hqla
    rsf_total = rsf_cat + rsf_hqla
    nsfr_pct = asf_total / rsf_total * 100 if rsf_total > 0 else float("inf")

    summary = pd.DataFrame(
        {
            "metric": ["LCR",   "NSFR"],
            "numerator_mb": [round(hqla, 1),      round(asf_total, 1)],
            "denominator_mb": [round(total_outflow, 1), round(rsf_total, 1)],
            "ratio_pct": [round(lcr_pct, 1),  round(nsfr_pct, 1)],
            "minimum_pct": [100, 100],
            "pass": [lcr_pct >= 100, nsfr_pct >= 100]
        }
    ).set_index("metric")

    return {
        "lcr": {
            "hqla": hqla,
            "total_outflow": total_outflow,
            "lcr_pct": lcr_pct,
            "min_liq_dynamic": min_liq_dynamic,
            "outflow_stable": outflow_stable,
            "outflow_unstable": outflow_unstable,
            "outflow_noncash": outflow_noncash
        },
        "nsfr": {
            "asf_total": asf_total,
            "rsf_total": rsf_total,
            "nsfr_pct": nsfr_pct,
            "asf_stable": asf_stable,
            "asf_unstable": asf_unstable,
            "asf_noncash": asf_noncash,
            "rsf_cat": rsf_cat,
            "rsf_hqla": rsf_hqla
        },
        "summary": summary
    }