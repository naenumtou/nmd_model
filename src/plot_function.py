
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.gridspec import GridSpec
from scipy import stats
from src.nmd_floor import price_nmd_floor

# Plot data for NMD
def plot_nmd_data(
    df: pd.DataFrame
) -> None:
    
    """
    Plot data for NMD.

    Description:
        Plot visualise all series on NMD Data.

    Args:
        data (pd.DataFrame): NMD Data input.

    Returns:
        Figure: Showing figure from matplotlib.
    """

    fig = plt.figure(figsize = (20, 14))
    gs  = GridSpec(3, 3, figure = fig)

    plot_specs = [
        #(row, col, columns_to_plot, ylabel, title)
        (0, 0, ["balance"], "MB", "Deposit Balance"),
        (0, 1, ["market_rate", "deposit_rate","repo_rate"], "Rate", "Interest Rates"),
        (0, 2, ["unemployment"], "Rate", "Unemployment Rate"),
        (1, 0, ["cds_spread"], "bps", "CDS Spread"),
        (1, 1, ["wealth"], "Index", "Client Wealth Index"),
        (1, 2, ["equity_return"], "Log Ret", "Equity Log-Return")
    ]

    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
    for row, col, cols, ylabel, title in plot_specs:
        ax = fig.add_subplot(gs[row, col])
        for i, c in enumerate(cols):
            ax.plot(
                df.index, df[c], color = colors[i % len(colors)],
                linewidth = 1.5, label = c
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        if len(cols) > 1:
            ax.legend(loc = "best")

    # Correlation heatmap
    ax_corr = fig.add_subplot(gs[2, :])
    corr = df.corr()
    im = ax_corr.imshow(corr, cmap = "RdBu_r", vmin = -1, vmax = 1, aspect = "auto")
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns)
    ax_corr.set_yticklabels(corr.columns)
    ax_corr.set_title("Correlation Matrix")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax_corr.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha = "center", va = "center", fontsize = 10,
                color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
            )
    plt.colorbar(im, ax = ax_corr, fraction = 0.02, pad = 0.02)

    fig.suptitle("NMD Synthetic Dataset", fontsize = 14, fontweight = "bold")
    plt.tight_layout()

    return plt.show()

# Plot all analysis for survival model
def plot_survival_results(
    cohort_matrix: pd.DataFrame,
    hazard_rates: pd.Series,
    worst_case: dict,
    reg_result: dict,
    mev_df: pd.DataFrame,
    mev_cols: list,
    n_cohorts_to_plot: int = 10
) -> None:

    """
    Visualise survival/decay model outputs across four panels.

    Description:
        Panel 1 — Cohort survival curves: balance decay over age for selected cohorts.
        Panel 2 — Snapshot hazard rates vs MEVs: time series comparison.
        Panel 3 — Hazard rate distribution with worst-case percentile marker.
        Panel 4 — MEV regression fit: actual vs fitted hazard rates.

    Args:
        cohort_matrix (pd.DataFrame)    : Cohort balance matrix.
        hazard_rates (pd.Series)        : Snapshot hazard rate series.
        worst_case (dict)               : Output from worst_case_runoff().
        reg_result (dict)               : Output from regress_runoff_on_mev().
        mev_df (pd.DataFrame)           : MEV DataFrame.
        mev_cols (list)                 : MEV column names used in regression.
        n_cohorts_to_plot (int)         : Number of cohort curves to show (default: 10).

    Returns:
        Figure: Showing figure from matplotlib.
    """

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle("Survival Decay Model Results", fontsize = 14, fontweight = "bold")
    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C"]

    # Panel 1: Cohort survival curves
    ax = axes[0, 0]
    step = max(1, cohort_matrix.shape[1] // n_cohorts_to_plot)
    sampled = cohort_matrix.iloc[:, ::step].dropna(how = "all")

    for i, col in enumerate(sampled.columns[:n_cohorts_to_plot]):
        cohort_data = sampled[col].dropna()
        if len(cohort_data) < 2:
            continue
        normalised = cohort_data / cohort_data.iloc[0] * 100
        ax.plot(
            range(len(normalised)), normalised.values,
            color=colors[i % len(colors)], alpha = 0.6, linewidth = 1.5
        )

    ax.axhline(50, color = "grey", linestyle = "--", linewidth = 0.8, label = "50% Survival")
    ax.axhline(30, color = "#DC2626", linestyle = "--", linewidth = 0.8, label = "30% Survival")
    ax.set_title("Cohort Survival Curves")
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("Surviving balance (%)")
    ax.legend(loc = "best")

    # Panel 2: Hazard rates vs MEVs
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    ax1.plot(
        hazard_rates.index, hazard_rates.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Hazard rate (Runoff %)"
    )
    ax1.set_ylabel("Hazard rate (%)")

    mev_colors = ["#DC2626", "#16A34A"]
    for i, col in enumerate(mev_cols):
        ax2.plot(
            mev_df.index, mev_df[col].values * 100,
            color = mev_colors[i % len(mev_colors)],
            linewidth = 1.0, linestyle = "--", label = col, alpha = 0.7
        )
    ax2.set_ylabel("MEV(s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.set_title("Hazard Rate and MEV(s)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # Panel 3: Hazard rate distribution
    ax = axes[1, 0]
    dist = worst_case["distribution"] * 100
    ax.hist(dist, bins = 25, color = "#2563EB", edgecolor="white", alpha = 0.55)
    ax.axvline(
        worst_case["worst_case_rate"] * 100, color = "#DC2626",
        linewidth = 2.0, linestyle = "--",
        label = f"P{worst_case["percentile"]:.0f} = {worst_case["worst_case_rate"]:.2%}"
    )
    ax.axvline(
        worst_case["mean_rate"] * 100, color = "#16A34A",
        linewidth = 2.5, label = f"Mean = {worst_case["mean_rate"]:.2%}"
    )
    ax.set_title("Hazard Rate Distribution")
    ax.set_xlabel("Runoff rate (%)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Panel 4: Regression fit
    ax = axes[1, 1]
    fitted = reg_result["fitted"]
    actual = hazard_rates.reindex(fitted.index)
    ax.plot(
        fitted.index, actual.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Hazard rate"
    )
    ax.plot(
        fitted.index, fitted.values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--", label = "Hazard rate (Fitted)"
    )
    ax.set_title("MEV Regression")
    ax.set_ylabel("Hazard rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()

    return plt.show()

# Plot all analysis for stable model
def plot_stable_results(
    balance: pd.Series,
    ci_result: dict,
    hp_result: dict,
    dd_monthly: dict,
    dd_yearly: dict,
    gbm_result: dict
) -> None:

    """
    Visualise stable/non-stable decomposition results across all four methods.

    Description:
        Panel 1 — CI Method: actual balance, fitted trend, and core (lower CI).
        Panel 2 — HP Filter: actual balance, HP trend, and core (lower bound).
        Panel 3 — GBM: log return distribution with shock threshold.
        Panel 4 — Drawdown: monthly and yearly drawdown time series.


    Args:
        balance (pd.Series)     : Aggregate deposit balance series.
        ci_result (dict)        : Output from stable_ci_method().
        hp_result (dict)        : Output from stable_hp_filter().
        dd_monthly (dict)       : Output from stable_drawdown(horizon='monthly').
        dd_yearly (dict)        : Output from stable_drawdown(horizon='yearly').
        gbm_result (dict)       : Output from stable_gbm().

    Returns:
        Figure: Showing figure from matplotlib.
    """

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle("Stable / Non-Stable Decomposition", fontsize = 14, fontweight = "bold")

    fmt = mdates.DateFormatter("%Y")
    loc = mdates.YearLocator(2)

    # Panel 1: CI Method
    ax = axes[0, 0]
    ax.plot(
        balance.index,
        balance.values,
        color = "#2563EB",
        linewidth = 1.5,
        label = "Deposit balance"
    )
    ax.plot(
        balance.index,
        ci_result["fitted"].values,
        color = "#EA580C",
        linewidth = 1.0,
        linestyle = "--",
        label = "Deposit balance (Fitted)"
    )
    ax.fill_between(
        balance.index,
        ci_result["core"].values,
        ci_result["fitted"].values,
        alpha = 0.20,
        color = "#DC2626",
        label = "Volatile"
    )
    ax.plot(
        balance.index,
        ci_result["core"].values,
        color = "#16A34A",
        linewidth = 1.5,
        label = f"Core (Stable {ci_result['stable_pct']:.2%})"
    )
    ax.set_title("Confidence Interval Method")
    ax.set_ylabel("Balance")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    # Panel 2: HP Filter
    ax = axes[0, 1]
    ax.plot(
        balance.index,
        balance.values,
        color = "#2563EB",
        linewidth = 1.5,
        label = "Deposit balance"
    )
    ax.plot(
        balance.index,
        hp_result["trend"].values,
        color = "#EA580C",
        linewidth = 1.0,
        linestyle = "--",
        label = f"Deposit balance (Fitted) (λ: {hp_result['lambda']:.0f})"
    )
    ax.fill_between(
        balance.index,
        hp_result["core"].values,
        hp_result["trend"].values,
        alpha = 0.20,
        color = "#DC2626",
        label = "Volatile"
    )
    ax.plot(
        balance.index,
        hp_result["core"].values,
        color = "#16A34A",
        linewidth = 1.5,
        label = f"Core (Stable {hp_result['stable_pct']:.2%})"
    )
    ax.set_title("HP Filter Method")
    ax.set_ylabel("Balance")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    # Panel 3: GBM log return distribution
    ax = axes[1, 0]
    log_ret = gbm_result["log_returns"] * 100
    ax.hist(
        log_ret,
        bins = 30,
        color = "#2563EB",
        alpha = 0.55,
        edgecolor = "white",
        density = True
    )

    x_range = np.linspace(log_ret.min(), log_ret.max(), 300)
    pdf = stats.norm.pdf(x_range, gbm_result["mu"] * 100, gbm_result["sigma"] * 100)
    ax.plot(
        x_range,
        pdf,
        color = "#EA580C",
        linewidth = 1.5,
        label = "Normal fit"
    )

    shock_pct = gbm_result["shock"] * 100
    ax.axvline(
        shock_pct,
        color = "#DC2626",
        linewidth = 2.5,
        linestyle = "--",
        label = f"Shock ({gbm_result['confidence']:.0%} CI): {shock_pct:.2f}%"
    )
    ax.set_title("GBM Model")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()

    # Panel 4: Drawdown
    ax = axes[1, 1]
    ax.plot(
        dd_monthly["drawdown"].index,
        dd_monthly["drawdown"].values * 100,
        color = "#2563EB",
        linewidth = 1.5,
        label = "Deposit balance (Monthly %Changed)"
    )
    ax.plot(
        dd_yearly["drawdown"].index,
        dd_yearly["drawdown"].values * 100,
        color = "#DC2626",
        linewidth = 1.0,
        linestyle = "--",
        label = "Deposit balance (Yearly %Changed)"
    )
    ax.axhline(
        dd_monthly["worst_drawdown"] * 100,
        color = "#2563EB",
        linewidth = 0.8,
        linestyle = ":",
        label = f"Worst monthly: {dd_monthly['worst_drawdown']:.2%}"
    )
    ax.axhline(0, color = "grey", linewidth = 0.6)
    ax.set_title("Drawdown Analysis")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()
    
    return plt.show()

# Plot all analysis for deposit rate model
def plot_deposit_rate_results(
    deposit_rate: pd.Series,
    market_rate: pd.Series,
    linear_result: dict,
    threshold_result: dict,
    jvd_result: dict
) -> None:

    """
    Visualise deposit rate model results across four panels.

    Description:
        Panel 1 — Linear model: actual deposit rate, market rate, and OLS fitted line.
        Panel 2 — Threshold model: actual vs fitted deposit rate with asymmetric dynamics.
        Panel 3 — JVD: actual vs fitted changes in deposit rate.
        Panel 4 — JVD level reconstruction: actual deposit rate vs JVD-implied level.

    Args:
        deposit_rate (pd.Series)    : Actual client deposit rate series.
        market_rate (pd.Series)     : Market rate series.
        linear_result (dict)        : Output from deposit_rate_linear().
        threshold_result (dict)     : Output from deposit_rate_threshold().
        jvd_result (dict)           : Output from deposit_rate_jvd().

    Returns:
        Figure: Showing figure from matplotlib.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deposit Rate Models — Results", fontsize = 14, fontweight = "bold")

    fmt = mdates.DateFormatter("%Y")
    loc = mdates.YearLocator(2)

    # Panel 1: Linear model
    ax = axes[0, 0]
    ax.plot(
        market_rate.index, market_rate.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Market rate"
    )
    ax.plot(
        deposit_rate.index, deposit_rate.values * 100,
        color = "#16A34A", linewidth = 1.5, label = "Deposit rate"
    )
    ax.plot(
        linear_result["fitted"].index,
        linear_result["fitted"].values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--",
        label = f"Deposit rate (Fitted)"
    )
    ax.set_title(
        f"Beta Regression | Beta: {linear_result['beta']:.4f}"
    )
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    # Panel 2: Threshold model
    ax = axes[0, 1]
    ax.plot(
        deposit_rate.index, deposit_rate.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Deposit rate"
    )
    ax.plot(
        threshold_result["fitted"].index,
        threshold_result["fitted"].values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--", label = "Deposit rate (Fitted)"
    )
    lam_up = threshold_result["lambda_up"]
    lam_down = threshold_result["lambda_down"]
    ax.set_title(
        f"Threshold Model | λ+: {lam_up:.4f} | λ-: {lam_down:.4f}"
    )
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    # Panel 3: JVD changes
    ax = axes[1, 0]
    delta_d = deposit_rate.diff().dropna()
    ax.plot(
        delta_d.index, delta_d.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Deposit rate (Changed)"
    )
    ax.plot(
        jvd_result["fitted"].index,
        jvd_result["fitted"].values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--", label = "Deposit rate (Changed - Fitted)"
    )
    ax.axhline(0, color = "grey", linewidth = 0.6)
    ax.set_title(
        f"JVD Model | β2: {jvd_result['beta_2']:.4f}"
    )
    ax.set_ylabel("Rate changed (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    # Panel 4: JVD level reconstruction
    ax = axes[1, 1]
    ax.plot(
        deposit_rate.index, deposit_rate.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Deposit rate"
    )
    ax.plot(
        jvd_result["fitted_level"].index,
        jvd_result["fitted_level"].values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--", label = "Deposit rate (Fitted)"
    )
    ax.set_title("JVD Prediction")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()

    return plt.show()

# Plot historical runoff proflie
def plot_hist_runoff(
    profile: dict,
    irrbb_buckets: pd.DataFrame
) -> None:

    """
    Visualise historical runoff proflie.

    Description:
        Panel 1 — Runoff profile: survival probability and monthly cash flows.
        Panel 2 — IRRBB buckets: balance distribution across repricing time buckets.

    Args:
        profile (dict)                  : Output from optimize_seed_rate()['profile'].
        irrbb_buckets (pd.DataFrame)    : Output from build_irrbb_buckets().

    Returns:
        Figure: Showing figure from matplotlib.
    """

    fig, axes = plt.subplots(2, 1, figsize = (7, 10))
    fig.suptitle("Historical runoff profile", fontsize = 14, fontweight = "bold")
    fmt = mdates.DateFormatter("%Y")
    loc = mdates.YearLocator(2)

    # Panel 1: Runoff profile
    ax = axes[0]
    months = np.arange(1, len(profile["survival_prob"]) + 1)
    ax2 = ax.twinx()
    ax.plot(
        months, profile["survival_prob"] * 100,
        color = "#2563EB", linewidth = 1.5, label = "Survival (%)"
    )
    ax2.bar(
        months, profile["cash_flows"] * 100,
        color = "#DC2626", alpha = 0.5, label = "Monthly runoff (%)"
    )
    ax.axvline(24, color = "grey", linewidth = 1.0, linestyle = "--", label = "Historical Ending")
    ax.set_title(f"Runoff Profile | WAL: {profile['wal_years']:.2f} years")
    ax.set_xlabel("Month")
    ax.set_ylabel("Survival (%)")
    ax2.set_ylabel("Monthly runoff (%)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    # Panel 2: IRRBB buckets
    ax = axes[1]
    buckets = irrbb_buckets.reset_index()
    bars = ax.bar(
        buckets["bucket"], buckets["balance_mb"],
        color = "#2563EB", alpha = 0.8, edgecolor = "white"
    )
    for bar, pct in zip(bars, buckets["pct_of_core"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.2f}%", ha = "center", va = "bottom"
        )
    ax.set_title("IRRBB Repricing Buckets")
    ax.set_ylabel("Balance (MB)")

    plt.tight_layout()
    
    return plt.show()

# Plot replicating portfolio
def plot_replicate_port(
    rep_result: dict,
    deposit_rate: pd.Series
) -> None:

    """
    Visualise replicating portfolio.

    Description:
        Panel 1 — Replicating portfolio weights per tenor.
        Panel 2 — Portfolio return vs actual deposit rate (tracking quality).

    Args:
        rep_result (dict)         : Output from static_replicating_portfolio().
        deposit_rate (pd.Series)  : Actual client deposit rate.

    Returns:
        None. Displays matplotlib figure.
    """

    fig, axes = plt.subplots(2, 1, figsize = (7, 10))
    fig.suptitle("Replicating Portfolio", fontsize = 14, fontweight = "bold")
    fmt = mdates.DateFormatter("%Y")
    loc = mdates.YearLocator(2)

    # Panel 1: Replicating portfolio
    ax = axes[0]
    weights = rep_result["weights"]
    bars = ax.bar(
        weights.index, weights.values * 100,
        color = "#16A34A", alpha = 0.8, edgecolor = "white"
    )
    for bar, val in zip(bars, weights.values):
        if val > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{val:.2%}",
                ha = "center", va = "bottom"
            )
    ax.set_title(f"Replicating Portfolio Weights | Margin = {rep_result['margin']:.2%}")
    ax.set_ylabel("Weight (%)")

    # Panel 2: Portfolio return
    ax = axes[1]
    port_ret = rep_result["portfolio_return"]
    dep_aligned = deposit_rate.reindex(port_ret.index)
    ax.plot(
        dep_aligned.index, dep_aligned.values * 100,
        color = "#2563EB", linewidth = 1.5, label = "Deposit rate"
    )
    ax.plot(
        port_ret.index, port_ret.values * 100,
        color = "#DC2626", linewidth = 1.0, linestyle = "--", label = "Portfolio return"
    )
    ax.set_title(f"Tracking Quality | TE = {rep_result['tracking_error']:.2%}")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()

    return plt.show()

# Plot EVE
def plot_economic_theory_results(
    paths_base: dict,
    paths_stressed: dict,
    eve_base: dict,
    eve_stressed: dict,
    delta_eve: float,
    n_plot_paths: int = 50
) -> None:

    """
    Visualise economic theory model results across four panels.

    Description:
        Panel 1 — Rate fan chart: base vs stressed simulated rate paths.
        Panel 2 — Mean cash flow: base vs stressed D × (r − d) over time.
        Panel 3 — PV distribution: histogram of path PVs for both scenarios.
        Panel 4 — ΔEVE summary: bar chart comparing base, stressed, and delta.

    Args:
        paths_base (dict)       : Output from simulate_paths() under base.
        paths_stressed (dict)   : Output from simulate_paths() under stress.
        eve_base (dict)         : Output from compute_eve() under base.
        eve_stressed (dict)     : Output from compute_eve() under stress.
        delta (float)           : Delta EVE.
        n_plot_paths (int)      : Number of paths to show in fan chart (default: 50).

    Returns:
        None. Displays matplotlib figure.
    """

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle("Economic Theory Model")
    months = np.arange(1, paths_base["r"].shape[1] + 1)

    # Panel 1: Rate fan chart
    ax = axes[0, 0]
    for i in range(min(n_plot_paths, paths_base["r"].shape[0])):
        ax.plot(months, paths_base["r"][i] * 100, color = "#2563EB", alpha = 0.25, linewidth = 0.5)
        ax.plot(months, paths_stressed["r"][i] * 100, color = "#DC2626", alpha = 0.25, linewidth = 0.5)
    ax.plot(months, paths_base["r"].mean(axis = 0) * 100,
            color = "#2563EB", linewidth = 1.5, label = "Base")
    ax.plot(months, paths_stressed["r"].mean(axis = 0) * 100,
            color = "#DC2626", linewidth = 1.5, label = "Stressed (+200bps)")
    ax.set_title("Simulated Rate Paths")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rate (%)")
    ax.legend()

    # Panel 2: Mean cash flow
    ax = axes[0, 1]
    ax.plot(months, eve_base["cf_mean"], color = "#2563EB", linewidth = 1.5, label = "Base")
    ax.plot(months, eve_stressed["cf_mean"], color = "#DC2626", linewidth = 1.5, linestyle = "--", label = "Stressed")
    ax.set_title("Mean Monthly Cash Flow")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash flow (MB)")
    ax.legend()

    # Panel 3: PV distribution
    ax = axes[1, 0]
    ax.hist(
        eve_base["pv_paths"], bins = 50, color = "#2563EB", alpha = 0.5, density = True,
        label = f"Base EVE: {eve_base['eve']:,.2f}"
    )
    ax.hist(
        eve_stressed["pv_paths"], bins = 50, color = "#DC2626", alpha = 0.4, density = True,
            label = f"Stressed EVE: {eve_stressed['eve']:,.2f}"
    )
    ax.axvline(eve_base["eve"], color = "#2563EB", linewidth = 1.5, linestyle = "--")
    ax.axvline(eve_stressed["eve"], color = "#DC2626", linewidth = 1.5, linestyle = "--")
    ax.set_title("Base and Stressed Distribution")
    ax.set_xlabel("PV of Benefits (MB)")
    ax.set_ylabel("Density")
    ax.legend()

    # Panel 4: ΔEVE summary
    ax = axes[1, 1]
    labels = ["Base EVE", "Stressed EVE", "ΔEVE"]
    values = [eve_base["eve"], eve_stressed["eve"], delta_eve]
    colors = ["#2563EB", "#DC2626",
              "#16A34A" if delta_eve >= 0 else "#EA580C"]
    bars = ax.bar(labels, values, color = colors, alpha = 0.8, edgecolor = "white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(max(values)) * 0.01,
            f"{val:,.0f}", ha = "center", va = "bottom"
        )
    ax.axhline(0, color = "grey", linewidth = 0.6)
    ax.set_title(
        f"ΔEVE: {delta_eve:,.2f} MB")
    ax.set_ylabel("Value (MB)")

    plt.tight_layout()
    
    return plt.show()

# Plot NMD Floor valuation
def plot_floor_results(
    bal_profile: dict,
    floor_result: dict,
    floor_result_atm: dict,
    forward_rates: np.ndarray,
    vol_info: dict,
    deposit_rate: pd.Series
) -> None:

    """
    Visualise NMD floor valuation results across four panels.

    Description:
        Panel 1 — Floorlet values per period (zero-strike vs ATM).
        Panel 2 — Floor value vs strike sensitivity.

    Args:
        bal_profile (dict)              : Input from runoff profile.
        floor_result (dict)             : Output from price_nmd_floor() at K=0.
        floor_result_atm (dict)         : Output from price_nmd_floor() at K=d0.
        forward_rates (np.ndarray)      : Forward rates per period.
        vol_info (dict)                 : Input from dynamics model.
        deposit_rate (pd.Series)        : Input from dynamics model.

    Returns:
        None. Displays matplotlib figure.
    """

    fig, axes = plt.subplots(2, 1, figsize = (7, 10))
    fig.suptitle("NMD Floor — Floorlet Valuation", fontsize = 14, fontweight = "bold")
    months = np.arange(1, 61)

    # Panel 1: Floorlet values per period
    ax = axes[0]
    ax.bar(
        months, floor_result["floorlet_values"],
        color = "#2563EB", alpha = 0.7, label = f"K: 0% (Floorlet value: {floor_result['floor_value']:,.2f} MB)"
    )
    ax.bar(
        months, floor_result_atm["floorlet_values"],
        color = "#DC2626", alpha = 0.5, label = f"K: d0 (Floorlet value: {floor_result_atm['floor_value']:,.2f} MB)"
    )
    ax.set_title("Floorlet Values per Period")
    ax.set_xlabel("Month")
    ax.set_ylabel("Floorlet value (MB)")
    ax.legend()

    # Panel 2: Floor value vs strike
    ax = axes[1]
    strikes = np.linspace(0.0, vol_info["r0"], 30)
    floor_vals = [
        price_nmd_floor(
            bal_profile["balance_profile"],
            forward_rates,
            vol_info["sigma_n"],
            vol_info["r0"],
            K
        )["floor_value"]
        for K in strikes
    ]
    ax.plot(
        strikes * 100, floor_vals,
        color = "#2563EB", linewidth = 1.5
    )
    ax.axvline(0, color = "grey", linewidth = 0.5, linestyle = "--", label = "K: 0% (Zero-floor)")
    ax.axvline(float(deposit_rate.iloc[-1]) * 100, color = "#DC2626", linewidth = 0.5, linestyle = "--", label = "K: d0 (Current deposit rate)")
    ax.set_title("Floor Value and Strike")
    ax.set_xlabel("Strike (%)")
    ax.set_ylabel("Floor value (MB)")
    ax.legend()


    plt.tight_layout()
    
    return plt.show()

# Plot visualise caterpillar
def plot_caterpillar_results(
    caterpillar: dict,
    opt_result: dict,
    yield_curve: pd.DataFrame
) -> None:

    """
    Visualise caterpillar structural hedge results across four panels.

    Description:
        Panel 1 — Tranche schedule: balance per tranche with duration bar.
        Panel 2 — Duration grid: feasible configurations by tenor and n_tranches.
        Panel 3 — Yield curve: current yield curve with target tenor marked.
        Panel 4 — NII sensitivity: ΔNII per rate shock scenario.

    Args:
        caterpillar (dict)          : Output from build_caterpillar() — best config.
        opt_result (dict)           : Output from optimise_caterpillar().
        yield_curve (pd.DataFrame)  : Output from build_yield_curve().

    Returns:
        None. Displays matplotlib figure.
    """

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle("Structural Hedge - Caterpillar", fontsize = 14, fontweight = "bold")

    # Panel 1: Tranche schedule
    ax = axes[0, 0]
    sched = caterpillar["schedule"]
    bars = ax.barh(
        sched["tranche"], sched["duration_yr"],
        color = "#2563EB", alpha = 0.8
    )
    ax.axvline(
        caterpillar["wal_liability"],
        color = "#DC2626", linewidth = 1.5, label = f"WAL:{caterpillar['wal_liability']:.2f} years"
    )
    ax.axvline(
        caterpillar["avg_duration"],
        color = "#16A34A", linewidth = 1.5, linestyle = "--", label = f"Average duration: {caterpillar['avg_duration']:.2f} years"
    )
    for bar, row in zip(bars, sched.itertuples()):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{row.balance_mb:,.0f} MB {row.yield_pct:.2f}%",
            va = "center"
        )
    ax.set_title(f"Tranche Schedule")
    ax.set_xlabel("Duration (years)")
    ax.legend()

    # Panel 2: Duration gap grid
    ax = axes[0, 1]
    grid = opt_result["grid"]
    for _, row in grid.iterrows():
        color = "#16A34A" if row["feasible"] else "#9A9A9A"
        ax.scatter(
            row["n_tranches"], row["tenor"], s = row["yield_pct"] * 30,
            color = color, alpha = 0.8
        )
        ax.text(
            row["n_tranches"] + 0.05, row["tenor"],
            f"{row['yield_pct']:.2f}%", va = "center"
        )
    ax.scatter([], [], color = "#16A34A", label = "Feasible")
    ax.scatter([], [], color = "#9A9A9A", label = "Infeasible")
    yticks = sorted(opt_result["grid"]["tenor"].unique())
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y)) for y in yticks])
    ax.set_title("Configuration Grid Yield")
    ax.set_xlabel("N Tranches")
    ax.set_ylabel("Tenor (years)")
    ax.legend()

    # Panel 3: Yield curve
    ax = axes[1, 0]
    tenors_plot = [0.25, 0.5, 1, 2, 3, 5, 10]
    current_yields = yield_curve.iloc[-1].values
    ax.plot(
        tenors_plot, current_yields * 100,
        color = "#2563EB", linewidth = 1.5, marker = "o", markersize = 5
    )
    ax.axvline(
        caterpillar["target_tenor"],
        color = "#16A34A", linewidth = 1.5, linestyle = "--",
        label = f"Target tenor: {caterpillar['target_tenor']} years"
    ) 
    ax.axvline(
        caterpillar["wal_liability"],
        color = "#DC2626", linewidth = 1.5,
        label = f"WAL: {caterpillar['wal_liability']:.2f} years"
    )
    ax.set_title("Current Yield Curve")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Yield (%)")
    ax.legend()

    # Panel 4: NII sensitivity
    ax = axes[1, 1]
    sensitivity = caterpillar["nii_sensitivity"]
    scenarios = list(sensitivity.keys())
    y1_vals = [v["delta_nii_year1_mb"] for v in sensitivity.values()]
    full_vals = [v["delta_nii_fullroll_mb"] for v in sensitivity.values()]
    x = np.arange(len(scenarios))
    ax.bar(
        x - 0.2, y1_vals, 0.4, label = "Year 1 (1 Tranche reprices)",
           color = "#2563EB", alpha = 0.8
    )
    ax.bar(
        x + 0.2, full_vals, 0.4, label = "Fully rolled (all tranches)",
           color = "#DC2626", alpha = 0.8
    )
    ax.axhline(0, color = "grey", linewidth = 0.5)
    ax.set_title("NII Sensitivity for Rate Scenario")
    ax.set_xlabel("Rate shock")
    ax.set_ylabel("ΔNII (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()

    plt.tight_layout()
    
    return plt.show()

# Plot visualise wealth allocation
def plot_allocation_results(
    alloc_result: dict,
    sensitivity_table: pd.DataFrame,
    yields: dict,
    deposit_rate: pd.Series
) -> None:

    """
    Visualise wealth allocation results across four panels.

    Description:
        Panel 1 — Optimal allocation: pie chart of weights.
        Panel 2 — NII breakdown: bar chart by strategy.
        Panel 3 — Sensitivity: NII vs EVE limit tradeoff.
        Panel 4 — Yield comparison: strategy yields vs deposit rate.

    Args:
        alloc_result (dict)                 : Output from optimise_allocation().
        sensitivity_table (pd.DataFrame)    : Output from allocation_sensitivity().
        yields (dict)                       : Output from get_strategy_yields().
        deposit_rate(pd.Series)             : Latest deposit rate.

    Returns:
        None. Displays matplotlib figure.
    """

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    fig.suptitle("Wealth Allocation Model", fontsize = 14, fontweight = "bold")

    w = alloc_result["weights"]
    nb = alloc_result["notional_mb"]
    nii_b = alloc_result["nii_breakdown"]

    # Panel 1: Allocation pie
    ax = axes[0, 0]
    labels = [
        f"Caterpillar\n{w['cat']:.2%} ({nb['cat']:,.2f} MB)",
        f"Floating\n{w['float']:.2%} ({nb['float']:,.2f} MB)",
        f"Liquidity Buffer\n{w['liq']:.2%} ({nb['liq']:,.2f} MB)"
    ]
    sizes = [w["cat"], w["float"], w["liq"]]
    colors = ["#2563EB", "#16A34A", "#F59E0B"]
    explode = (0.03, 0.03, 0.03)
    ax.pie(
        sizes,
        labels = labels,
        colors = colors, explode = explode,
        autopct = "%1.2f%%", startangle = 90
    )
    ax.set_title(
        f"Optimal Allocation | NII = {alloc_result['nii_annual']:,.2f} MB"
    )

    # Panel 2: NII breakdown
    ax = axes[0, 1]
    strategies = ["Caterpillar", "Floating", "Liquidity"]
    nii_vals = [nii_b["cat"], nii_b["float"], nii_b["liq"]]
    bars = ax.bar(
        strategies, nii_vals,
        color = ["#2563EB", "#16A34A", "#F59E0B"], alpha = 0.8
    )
    for bar, val in zip(bars, nii_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:,.2f} MB", ha = "center", va = "bottom"
        )
    ax.set_title("NII Breakdown per Strategy")
    ax.set_ylabel("NII (MB)")

    # Panel 3: NII vs EVE limit sensitivity
    ax = axes[1, 0]
    st = sensitivity_table.reset_index()
    ax2 = ax.twinx()
    ax.plot(
        st["eve_limit_mb"], st["nii_annual_mb"],
        color = "#2563EB", linewidth = 2, marker = "o", label = "NII (MB)"
    )
    ax2.plot(
        st["eve_limit_mb"], st["w_cat"],
        color = "#DC2626", linewidth = 1.5, linestyle = "--", label = "Weights Caterpillar (%)"
    )
    ax.set_title("NII vs EVE Limit Tradeoff")
    ax.set_xlabel("EVE Limit (MB)")
    ax.set_ylabel("NII (MB)")
    ax2.set_ylabel("Caterpillar weight (%)")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2)

    # Panel 4: Yield comparison
    ax = axes[1, 1]
    yc = yields["yield_curve"]
    tenor_vals = [0.25, 0.5, 1, 2, 3, 5, 10]
    ax.plot(
        tenor_vals, yc.values * 100,
        color = "#2563EB", linewidth = 1.5, marker = "o", label = "Yield curve"
    )
    ax.axhline(
        yields["cat"] * 100, color = "#2563EB",
        linewidth = 0.5, linestyle = "--",
        label = f"Caterpillar yield: {yields['cat']:.2%}"
    )
    ax.axhline(
        yields["float"] * 100, color = "#16A34A",
        linewidth = 0.5, linestyle = "--",
        label = f"Floating yield: {yields['float']:.2%}"
    )
    ax.axhline(
        float(deposit_rate.iloc[-1]) * 100, color = "#DC2626",
        linewidth = 1.0, linestyle = "--",
        label = f"Deposit rate: {float(deposit_rate.iloc[-1]):.2%}"
    )
    ax.set_title("Strategy Yields and Deposit Rate")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Yield (%)")
    ax.legend()

    plt.tight_layout()
    
    return plt.show()

# Plot visualise IRRBB Integration
def plot_irrbb_results(
    report: dict,
    nb05: dict,
    nb06: dict
) -> None:

    """
    Visualise IRRBB integration results across four panels.

    Description:
        Panel 1 — Repricing gap: balance distribution across IRRBB buckets.
        Panel 2 — EVE sensitivity: bar chart per rate shock scenario.
        Panel 3 — NII sensitivity: ΔNII year 1 vs fully rolled.

    Args:
        report (dict) : Output from build_irrbb_report().
        nb05   (dict) : Output from reproduce_nb05().
        nb06   (dict) : Output from reproduce_nb06().

    Returns:
        None. Displays matplotlib figure.
    """

    fig = plt.figure(figsize = (14, 10))
    fig.suptitle("IRRBB Integration Report", fontsize = 14, fontweight = "bold")

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel 1: Repricing gap
    ax = ax1
    gap = report["repricing_gap"].reset_index()
    colors = ["#DC2626"] + ["#2563EB"] * (len(gap) - 1)
    bars = ax.bar(
        gap["bucket"],gap["balance_mb"],
        color = colors, alpha = 0.8
    )
    for bar, val in zip(bars, gap["balance_mb"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val:,.0f}", ha = "center", va = "bottom"
        )
    ax.set_title("IRRBB Repricing Gap (MB)")
    ax.set_ylabel("Balance (MB)")

    # Panel 2: EVE sensitivity
    ax = ax2
    eve_t = report["eve_table"].reset_index()
    colors2 = ["#2563EB" if s == "Base" else ("#DC2626" if float(v.replace("%","")) < 0 else "#16A34A")
               for s, v in zip(eve_t["scenario"], eve_t["ΔEVE (%)"].fillna("0%"))]
    ax.bar(
        eve_t["scenario"], eve_t["EVE (MB)"],
        color = colors2, alpha = 0.8
    )
    ax.axhline(nb06["eve_base"], color = "grey", linewidth = 0.5, linestyle = "--")
    ax.set_title("EVE per Rate Scenario (MB)")
    ax.set_ylabel("EVE (MB)")

    # Panel 3: NII sensitivity
    ax = ax3
    nii_t = report["nii_table"].reset_index()
    x = np.arange(len(nii_t))
    ax.bar(
        x - 0.2, nii_t["ΔNII year 1 (MB)"], 0.4,
        label = "Year 1", color = "#2563EB", alpha = 0.8
    )
    ax.bar(
        x + 0.2, nii_t["ΔNII fully rolled (MB)"], 0.4,
        label = "Fully rolled", color = "#DC2626", alpha = 0.8
    )
    ax.axhline(0, color = "grey", linewidth = 0.5)
    ax.set_title("NII Sensitivity per Scenario (MB)")
    ax.set_ylabel("ΔNII (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(nii_t["scenario"])
    ax.legend()


    plt.tight_layout()
    return plt.show()