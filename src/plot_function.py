
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter

# Plot assets return
def plot_return(
    data: pd.DataFrame
) -> None:
    
    """
    Plot assets return.

    Description:
        Plot assets return from wealth data.

    Args:
        data (pd.DataFrame): Data input for assets return.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax = plt.subplots(figsize = (10, 6))

    # Return
    cols_line = ["r_deposit", "r_bond", "r_equity", "r_fund"]
    for col in cols_line:
        ax.plot(data.index, data[col], label = col)
    ax.set_ylabel("Return")
    ax.legend(loc = "upper left")
    plt.title("Assets return")
    plt.tight_layout()

    return plt.show()

# Plot assets allocation
def plot_allocation(
    data: pd.DataFrame
) -> None:
    
    """
    Plot assets return.

    Description:
        Plot assets return from wealth data.

    Args:
        data (pd.DataFrame): Data input for assets allocation.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax1 = plt.subplots(figsize = (10, 6))

    # Allocation
    cols = ["share_deposit", "share_bond", "share_equity", "share_fund"]    
    colors = ["#4C72B0", "#808080", "#C44E52", "#8172B2"]
    ax1.stackplot(data.index, [data[c] for c in cols], labels = cols, colors = colors, alpha = 0.8)
    ax1.yaxis.set_major_formatter(PercentFormatter(1, decimals = 2))
    ax1.set_xlim(data.index.min(), data.index.max())
    ax1.set_ylabel("Portfolio shared")
    ax1.set_ylim(0, 1)
    ax1.legend(loc = "upper left")

    # Deposit balance
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["deposit_balance"], color = "#ffd500", linewidth = 2)
    ax2.set_ylabel("Deposit balance")

    plt.title("Portfolio allocation with deposit balance")
    plt.tight_layout()

    return plt.show()

# Plot projection
def plot_project(
    data: pd.DataFrame
) -> None:
    
    """
    Plot projection on deposit balance.

    Description:
        Plot projection on deposit balance.

    Args:
        data (pd.DataFrame): Data input for projection on deposit balance.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    x = data["deposit_balance"]
    y = data["total_wealth"]

    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, color = "gray")

    # Regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color = "#ffd500")

    plt.xlabel("Deposit balance")
    plt.ylabel("Total wealth")
    plt.title("Deposit and wealth projection")
    plt.tight_layout()

    return plt.show()


