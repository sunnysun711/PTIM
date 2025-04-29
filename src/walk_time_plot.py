"""
This Python module is designed to visualize and analyze egress and transfer time 
distributions.

Data sources:
- Egress time data (`eg_t`): This DataFrame contains egress time information for 
    various physical platforms.
- Transfer time data (`tr_t`): This DataFrame contains transfer time information 
    for various transfer links.
- Physical platform data (`platforms`): This array contains information about the
    physical platforms. (Generated in `scripts/prep_network.py`)

Dependencies:
- `numpy`: For numerical operations and array manipulations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For creating visualizations.
- `seaborn`: For enhanced statistical graphics.

- `src.utils.ts2tstr`: A custom utility function for converting timestamps to 
    time strings.
- `src.walk_time_dis_fit.fit_pdf_cdf`: A custom function for fitting probability 
    density and cumulative distribution functions.
- `src.walk_time_dis_fit.reject_outlier_bd`: A custom function for rejecting 
    outliers based on a given method.

Usage:
Import and call the following functions as needed:
- `plot_egress_all()`: Generate and save egress time distribution plots.
- `plot_transfer_all()`: Generate and save transfer time distribution plots.
- `_plot_walk_time_dis()`: (protected) The base method for visualizing walk time 
    distribution through a composite plot (only for one data one plot).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from src import config
from src.utils import ts2tstr
from src.globals import get_platform
from src.walk_time_dis_fit import fit_pdf_cdf, reject_outlier_bd


def _plot_walk_time_dis(
        walk_time: np.ndarray, alight_ts: np.ndarray, title: str = "", show_: bool = True, fit_curves: list[str] = None,
) -> None | plt.Figure:
    """
    Visualize walk time distribution through a composite plot containing:
    - Scatter plot of walk times vs alighting times
    - Histogram of walk time distribution
    - Boxplot of walk times by time bins

    Parameters:
        `walk_time` (np.ndarray): Array of walk times in seconds
        `alight_ts` (np.ndarray): Array of alighting timestamps in seconds since midnight
        `title` (str): Additional title text for the plot
        `show_` (bool): If True, displays the plot; if False, return the figure object
        `fit_curves`: List of curve names to fit and plot, e.g., ["kde", "gamma", "lognorm"].
            If None, all three fitting will be done.

    Returns:
        None or plt.Figure: If show_ is True, returns None; otherwise, returns the figure object
    """
    if show_:
        matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
    else:
        matplotlib.use("Agg")  # Use non-interactive backend for saving figures

    # Set up the figure and axes for the grid layout
    # Creates a 2x2 grid with:
    # - Top-left: scatter plot (80% width)
    # - Top-right: histogram (20% width)
    # - Bottom-left: boxplot (full width)
    fig = plt.figure(figsize=(10, 6))
    grid = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 0.25])
    fit_curves = ["kde", "gamma",
                  "lognorm"] if fit_curves is None else fit_curves

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


def plot_egress_all(
        eg_t: pd.DataFrame,
        save_subfolder: str = "",
        save_on: bool = True,
):
    """
    Generates and saves egress time distribution plots for all physical platform IDs, including scatter plots,
    histograms, and boxplots. The plots visualize the relationship between egress times and alighting
    timestamps, with outliers rejected based on z-score method.

    This function processes the egress time data for different physical platform IDs, applies outlier rejection
    to the egress time data, and generates a plot for each platform. The plots are saved as PDF and PNG
    files to the specified directory.
    
    Parameters:
    ------
    :param save_subfolder: The subfolder where plots will be saved.
        If not specified, plots are saved in the default figure folder.

    :param eg_t: DataFrame containing egress time data.
        The DataFrame should have the following columns:
            - 'physical_platform_id': Physical platform ID.
            - 'egress_time': Egress time in seconds.
            - 'alight_ts': Alighting timestamp in seconds since midnight.

    :param save_on: If True, saves the plots to PDF and PNG files. 
        If False, shows the plots interactively.

    Returns:
    ------
    None
    """
    platforms = get_platform()  # pp_id, node_id, uid

    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print(f"[INFO] Plotting ETD...")
    for uid in range(1001, 1137):
        all_pp_this_uid = np.unique(platforms[platforms[:, 2] == uid][:, 0])
        for pp_id in all_pp_this_uid:
            et = eg_t[eg_t["physical_platform_id"] == pp_id]
            node_ids = [str(int(i))
                        for i in platforms[platforms[:, 0] == pp_id][:, 1]]
            if et.shape[0] == 0:
                # pp_id: 103803 (Line 10 Down) no egress time data. Thus will be skipped here.
                # FIXME: this is probably not the best solution. (Now will be handled when calculating.)
                # (when indexing 103803, 103804 will be used.)
                print(f"\n[WARNING] No egress data for pp_id: {pp_id}.\n")
                continue

            title = f"(Egress): {pp_id}_{'-'.join(node_ids)}"
            raw_size = et.shape[0]
            lb, ub = reject_outlier_bd(
                data=et['egress_time'].values, method="zscore", abs_max=500)
            et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
            print(
                f"{title} | Data size: {et.shape[0]} / {raw_size} | BD: [{lb:.4f}, {ub:.4f}]")
            fig = _plot_walk_time_dis(
                walk_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=title,
                show_=not save_on  # return figure if not showing; return None if showing
            )
            if fig is not None:  # Save the plot to PDF file and a small png file for preview
                fig.savefig(
                    fname=f"{saving_dir}/ETD_{title[10:]}.pdf", dpi=600)
                fig.savefig(
                    fname=f"{saving_dir}/ETD_{title[10:]}.png", dpi=100)
                plt.close(fig)
    return


def plot_transfer_all(
    tr_t: pd.DataFrame,
    save_subfolder: str = "",
    save_on: bool = True,
):
    """
    Generates and saves transfer time distribution plots for all transfer links, 
    which means a tuple of pp_ids. The plots include scatter plots, histograms, 
    and box plots. The plots visualize the relationship between transfer times 
    and alighting timestamps, with outliers rejected based on z-score method.

    Note: The transfer from `pp_id1` to `pp_id2` is considered equivalent to 
    the transfer from `pp_id2` to `pp_id1`. Therefore, the final representation 
    of a transfer link will be a tuple of physical platform IDs 
    (`pp_id1`, `pp_id2`), with the smaller ID placed first.

    Parameters:
    ------
    :param tr_t: DataFrame containing transfer time data.
        The DataFrame should have the following columns:
            - 'path_id': Path ID.
            - 'seg_id': Segment ID.
            - 'pp_id1': Physical platform ID of the first platform.
            - 'pp_id2': Physical platform ID of the second platform.
            - 'alight_ts': Alighting timestamp in seconds since midnight.
            - 'transfer_time': Transfer time in seconds.
            - 'transfer_type': Type of transfer ('egress-entry', 'platform_swap').

    :param save_subfolder: The subfolder where plots will be saved.
        If not specified, plots are saved in the default figure folder.

    :param save_on: If True, saves the plots to PDF and PNG files. 
        If False, shows the plots interactively.
    """
    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print("[INFO] Plotting TTD...")
    tr_t["pp_id_min"] = tr_t[["pp_id1", "pp_id2"]].min(axis=1)
    tr_t["pp_id_max"] = tr_t[["pp_id1", "pp_id2"]].max(axis=1)

    # egress-entry transfers
    for (pp_id_min, pp_id_max), df_ in tr_t[tr_t["transfer_type"] == "egress-entry"].groupby(
            ["pp_id_min", "pp_id_max"])[["transfer_time", "alight_ts"]]:
        title = f"(Transfer egress-entry): {pp_id_min}-{pp_id_max}"
        raw_size = df_.shape[0]
        data_bd = reject_outlier_bd(
            data=df_["transfer_time"].values, method="zscore", abs_max=500)
        df_ = df_[(df_["transfer_time"] >= data_bd[0]) &
                  (df_["transfer_time"] <= data_bd[1])]
        print(
            f"{title} | Data size: {df_.shape[0]} / {raw_size} | BD: [{data_bd[0]:.4f}, {data_bd[1]:.4f}]")

        fig = _plot_walk_time_dis(
            walk_time=df_["transfer_time"], alight_ts=df_[
                "alight_ts"].values, title=title,
            show_=not save_on,  # return figure if not showing; return None if showing
            fit_curves=None,  # ["kde", "gamma", "lognorm"]
        )
        if fig is not None:  # Save the plot to PDF file and a small png file for preview
            fig.savefig(fname=f"{saving_dir}/TTD_EE_{title[25:]}.pdf", dpi=600)
            fig.savefig(fname=f"{saving_dir}/TTD_EE_{title[25:]}.png", dpi=100)
            plt.close(fig)

    # platform-swap transfers
    for (pp_id_min, pp_id_max), df_ in tr_t[tr_t["transfer_type"] == "platform_swap"].groupby(
            ["pp_id_min", "pp_id_max"])[["transfer_time", "alight_ts"]]:
        title = f"(Transfer platform_swap): {pp_id_min}-{pp_id_max}"
        raw_size = df_.shape[0]
        data_bd = reject_outlier_bd(
            data=df_["transfer_time"].values, method="zscore", abs_max=500)
        df_ = df_[(df_["transfer_time"] >= data_bd[0]) &
                  (df_["transfer_time"] <= data_bd[1])]
        print(
            f"{title} | Data size: {df_.shape[0]} / {raw_size} | BD: [{data_bd[0]}, {data_bd[1]}]")

        fig = _plot_walk_time_dis(
            walk_time=df_['transfer_time'], alight_ts=df_[
                "alight_ts"].values, title=title,
            show_=not save_on,  # return figure if not showing; return None if showing
            fit_curves=[],  # ["kde", "gamma", "lognorm"]
        )
        if fig is not None:  # Save the plot to PDF file and a small png file for preview
            fig.savefig(fname=f"{saving_dir}/TTD_PS_{title[26:]}.pdf", dpi=600)
            fig.savefig(fname=f"{saving_dir}/TTD_PS_{title[26:]}.png", dpi=100)
            plt.close(fig)
    return
