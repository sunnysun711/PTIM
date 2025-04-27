"""
This script analyzes walking time distributions at each station UID
based on assigned and candidate feasible itineraries.

Data sources:
- 'feas_iti_assigned.pkl': assigned itineraries (only one per passenger).
- 'feas_iti_left.pkl': itineraries sharing the same alighting train but not selected.

Main steps:
1. Combine walking time samples from both files by station UID.
2. Compute the probability density function (PDF) and cumulative distribution function (CDF).
3. Save the statistical results to a CSV file for further analysis or visualization.

Usage:
Import and call the following functions as needed:
- `get_pdf()`: Calculate PDF from walking time samples.
- `get_cdf()`: Calculate CDF from walking time samples.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

from src.globals import get_platform

matplotlib.use("Agg")  # Use non-interactive backend for saving figures
# matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting

from matplotlib import pyplot as plt

from src import config
from src.utils import ts2tstr
from src.walk_time_dis_fit import fit_pdf_cdf, reject_outlier_bd



def plot_walk_time_dis(
        walk_time: np.ndarray, alight_ts: np.ndarray, title: str = "", show_: bool = True, fit_curves: list[str] = None,
) -> None | plt.Figure:
    """
    Visualize walk time distribution through a composite plot containing:
    - Scatter plot of walk times vs alighting times
    - Histogram of walk time distribution
    - Boxplot of walk times by time bins

    Parameters:
        walk_time (np.ndarray): Array of walk times in seconds
        alight_ts (np.ndarray): Array of alighting timestamps in seconds since midnight
        title (str): Additional title text for the plot
        show_ (bool): If True, displays the plot; if False, return the figure object
        fit_curves: List of curve names to fit and plot, e.g., ["kde", "gamma", "lognorm"].
            If None, all three fitting will be done.

    Returns:
        None or plt.Figure: If show_ is True, returns None; otherwise, returns the figure object
    """
    # Set up the figure and axes for the grid layout
    # Creates a 2x2 grid with:
    # - Top-left: scatter plot (80% width)
    # - Top-right: histogram (20% width)
    # - Bottom-left: boxplot (full width)
    fig = plt.figure(figsize=(10, 6))
    grid = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 0.25])
    fit_curves = ["kde", "gamma", "lognorm"] if fit_curves is None else fit_curves

    # Histogram of the walk time distribution (all day)
    ax_right = fig.add_subplot(grid[0, 1])
    # check usage: https://seaborn.pydata.org/generated/seaborn.histplot.html
    n_bins = 30
    bin_width = (walk_time.max() - walk_time.min()) / n_bins
    scale = bin_width * walk_time.size
    sns.histplot(
        y=walk_time, kde=False, ax=ax_right, color="blue", alpha=0.3, bins=n_bins, stat='count', element="bars")

    x_values = np.linspace(0, 500, 501)
    if "kde" in fit_curves:
        # Fit KDE and plot it on the histogram
        kde_pdf_f, _ = fit_pdf_cdf(walk_time, method="kde")
        ax_right.plot(kde_pdf_f(x_values) * scale, x_values,
                      color="red", label="KDE Fit", lw=1)
    if "gamma" in fit_curves:
        # Fit Gamma and LogNormal distributions and plot them on the histogram
        gamma_pdf_f, _ = fit_pdf_cdf(walk_time, method="gamma")
        ax_right.plot(gamma_pdf_f(x_values) * scale, x_values,
                      color="green", label="Gamma Fit", lw=1)

    if "lognorm" in fit_curves:
        lognorm_pdf_f, _ = fit_pdf_cdf(walk_time, method="lognorm")
        ax_right.plot(lognorm_pdf_f(x_values) * scale, x_values,
                      color="orange", label="LogNormal Fit", lw=1)

    ax_right.set_xlabel("Frequency")
    ax_right.set_ylabel("")
    if fit_curves:
        ax_right.legend()

    # Scatter plot showing relationship between alighting time and walk time
    ax_main = fig.add_subplot(grid[0, 0], sharey=ax_right)
    ax_main.scatter(alight_ts, walk_time, alpha=0.4)
    # ax_scatter.set_xlabel("Alight Timestamp")
    ax_main.set_ylabel("Walk Time")
    # Set x-axis ticks to show hourly labels from 6:00 to 24:00
    ax_main.set_xticks(range(6 * 3600, 24 * 3600 + 1, 3600))
    ax_main.set_xticklabels([f"{i:02}" for i in range(6, 25, 1)])
    ax_main.set_xlim(6 * 3600, 24 * 3600)
    ax_main.set_ylim(walk_time.min() - 10, walk_time.max() + 10)
    ax_main.set_title("Walk Time Distribution " + title)
    ax_main.set_xlabel("Alight Timestamp (Hour)")

    # Boxplot of egress time versus alight_ts, with customized bin width
    _bin_width = 1800
    alight_ts_binned = (alight_ts // _bin_width) * _bin_width

    ax_bottom = fig.add_subplot(grid[1, 0])
    sns.boxplot(x=alight_ts_binned, y=walk_time, ax=ax_bottom)
    ax_bottom.set_xlabel(f"Alight Timestamp")
    ax_bottom.set_ylabel("Walk Time")
    # Set x-axis ticks to show 30-minute intervals
    ax_bottom.set_xticks(
        [i - 0.5 for i in range((24 - 6) * 3600 // _bin_width + 1)])
    ax_bottom.set_xticklabels(
        [ts2tstr(ts) for ts in range(6 * 3600, 24 * 3600 + 1, _bin_width)])
    ax_bottom.set_xlim(- 0.5, (24 - 6) * 3600 // _bin_width - 0.5)
    # Rotate x-axis labels for better readability
    for label in ax_bottom.get_xticklabels():
        label.set_rotation(90)

    plt.tight_layout()
    if show_:  # Show the plot interactively
        plt.show()
        return None
    else:
        return fig


def _plot_egress_time_dis_all(
        eg_t: pd.DataFrame,
        save_subfolder: str = "",
        save_on: bool = True,
):
    """"""
    platforms = get_platform()  # pp_id, node_id, uid

    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    print(f"[INFO] Plotting ETD...")
    for uid in range(1001, 1137):
        _pl_info = platforms[platforms[:, 2] == uid]
        for pl_id in np.unique(_pl_info[:, 0]):
            plat_ids = _pl_info[_pl_info[:, 0] == pl_id][:, 1]
            



def plot_egress_time_dis_all(
        et_: pd.DataFrame,
        physical_link_info: np.ndarray,
        save_subfolder: str = "",
        save_on: bool = True,
):
    """
    Generates and saves egress time distribution plots for all platform links, including scatter plots,
    histograms, and boxplots. The plots visualize the relationship between egress times and alighting
    timestamps, with outliers rejected based on z-score method.

    This function processes the egress time data for different platform links, applies outlier rejection
    to the egress time data, and generates a plot for each platform. The plots are saved as PDF and PNG
    files to the specified directory.

    Parameters:
    ------
    :param save_subfolder: The subfolder where plots will be saved.
        If not specified, plots are saved in the default figure folder.

    :param et_: DataFrame containing egress time data.
        The DataFrame should have the following columns:
            - 'node2': The station UID of the egress path.
            - 'node1': The platform node_id of the egress path.

    :param physical_link_info: Numpy array with `pl_id`, `platform_id`, `uid`.

    :param save_on: If True, the plots are saved to files. If False, the plots are displayed.

    """
    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print(f"[INFO] Plotting ETD...")
    for uid in range(1001, 1137):
        _pl_info = physical_link_info[physical_link_info[:, 2] == uid]
        for pl_id in np.unique(_pl_info[:, 0]):
            plat_ids = _pl_info[_pl_info[:, 0] == pl_id][:, 1]
            et = et_[et_["node1"].isin(plat_ids)]
            if et.shape[0] == 0:
                continue
            title = f" - Egress {uid}_{plat_ids}"
            raw_size = et.shape[0]
            lb, ub = reject_outlier_bd(
                data=et['egress_time'].values, method="zscore", abs_max=500)
            et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
            print(
                f"{title[3:]} | Data size: {et.shape[0]} / {raw_size} | BD: [{lb:.4f}, {ub:.4f}]")

            fig = plot_walk_time_dis(
                walk_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=title,
                show_=not save_on  # return figure if not showing; return None if showing
            )
            if fig is not None:  # Save the plot to PDF file and a small png file for preview
                fig.savefig(fname=f"{saving_dir}/ETD_{title[10:]}.pdf", dpi=600)
                fig.savefig(fname=f"{saving_dir}/ETD_{title[10:]}.png", dpi=100)
                plt.close(fig)
    return


def plot_transfer_time_dis_all(
        df_tt: pd.DataFrame,
        df_ps2t: pd.DataFrame,
        save_subfolder: str = "",
        save_on: bool = True,
):
    """
    Generates and saves transfer time distribution plots for all transfer links, including scatter plots,
    histograms, and box plots. The plots visualize the relationship between transfer times and alighting
    timestamps, with outliers rejected based on z-score method.

    The plots are saved as PDF and PNG files to the specified directory.
    Parameters:
    ------
    :param df_tt: Optional DataFrame containing transfer time data.

    :param df_ps2t: Optional DataFrame mapping path seg info to transfer info.

    :param save_subfolder: The subfolder where plots will be saved.
        If not specified, plots are saved in the default figure folder.
    :param save_on: If True, the plots are saved to files. If False, the plots are displayed.
    """
    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print("[INFO] Plotting TTD...")
    df = pd.merge(df_tt, df_ps2t, on=["path_id", "seg_id"], how="left")

    # egress-entry transfers
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "egress-entry"].groupby(
            ["p_uid_min", "p_uid_max"])[["transfer_time", "alight_ts"]]:
        title = f" - Transfer egress-entry {p_uid1}-{p_uid2}"
        raw_size = df_.shape[0]
        data_bd = reject_outlier_bd(
            data=df_["transfer_time"].values, method="zscore", abs_max=500)
        df_ = df_[(df_["transfer_time"] >= data_bd[0]) & (df_["transfer_time"] <= data_bd[1])]
        print(f"{title[3:]} | Data size: {df_.shape[0]} / {raw_size} | BD: [{data_bd[0]}, {data_bd[1]}]")

        fig = plot_walk_time_dis(
            walk_time=df_["transfer_time"], alight_ts=df_["alight_ts"].values, title=title,
            show_=not save_on,  # return figure if not showing; return None if showing
            fit_curves=None,  # ["kde", "gamma", "lognorm"]
        )
        if fig is not None:  # Save the plot to PDF file and a small png file for preview
            fig.savefig(fname=f"{saving_dir}/TTD_{title[12:]}.pdf", dpi=600)
            fig.savefig(fname=f"{saving_dir}/TTD_{title[12:]}.png", dpi=100)
            plt.close(fig)

    # platform-swap transfers
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "platform_swap"].groupby(
            ["p_uid_min", "p_uid_max"])[["transfer_time", "alight_ts"]]:
        title = f" - Transfer platform_swap {p_uid1}-{p_uid2}"
        raw_size = df_.shape[0]
        data_bd = reject_outlier_bd(
            data=df_["transfer_time"].values, method="zscore", abs_max=500)
        df_ = df_[(df_["transfer_time"] >= data_bd[0]) & (df_["transfer_time"] <= data_bd[1])]
        print(f"{title[3:]} | Data size: {df_.shape[0]} / {raw_size} | BD: [{data_bd[0]}, {data_bd[1]}]")

        fig = plot_walk_time_dis(
            walk_time=df_['transfer_time'], alight_ts=df_["alight_ts"].values, title=title,
            show_=not save_on,  # return figure if not showing; return None if showing
            fit_curves=[],  # ["kde", "gamma", "lognorm"]
        )
        if fig is not None:  # Save the plot to PDF file and a small png file for preview
            fig.savefig(fname=f"{saving_dir}/TTD_{title[12:]}.pdf", dpi=600)
            fig.savefig(fname=f"{saving_dir}/TTD_{title[12:]}.png", dpi=100)
            plt.close(fig)
    return
