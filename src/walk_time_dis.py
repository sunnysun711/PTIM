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
from src import config
from src.utils import read_, read_all, ts2tstr
from src.globals import get_afc, get_k_pv, get_pl_info, get_etd, get_link_info, get_platform
from tqdm import tqdm
from scipy.stats import kstest
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import combinations
import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')


def filter_egress_time_from_left() -> pd.DataFrame:
    """
    Find rids in left.pkl where all feasible itineraries share the same final train_id.

    :return: A DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
             It includes the following columns:
             - "node1": The starting node of the egress path.
             - "node2": The ending node of the egress path.
             - "alight_ts": The time when the passenger alighted from the vehicle.
             - "ts2": The time when the passenger exited the station.
             - "egress_time": The calculated egress time, which is the difference between "ts2" and "alight_ts".
    """
    df = read_(config.CONFIG["results"]["left"], show_timer=False)

    # Keep only the last segment for each itinerary of one rid
    last_seg_all_iti = df.groupby(["rid", "iti_id"]).last().reset_index()

    # filter rids with same train_ids in last seg (Line 7 issue is resolved by default)
    unique_end_count = last_seg_all_iti.groupby("rid")["train_id"].nunique()
    same_train_rids = unique_end_count[unique_end_count == 1].index

    # get last in_vehicle link
    last_seg = last_seg_all_iti[
        last_seg_all_iti["rid"].isin(same_train_rids)
    ].groupby("rid").last().drop(columns=["iti_id", "board_ts"])

    return calculate_egress_time(last_seg)


def filter_egress_time_from_assigned() -> pd.DataFrame:
    """
    Find rids in all assigned_*.pkl files where all feasible itineraries share the same final train_id.

    :return: A DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
             It includes the following columns:
             - "node1": The starting node of the egress path.
             - "node2": The ending node of the egress path.
             - "alight_ts": The time when the passenger alighted from the vehicle.
             - "ts2": The time when the passenger exited the station.
             - "egress_time": The calculated egress time, which is the difference between "ts2" and "alight_ts".
    """
    df = read_all(config.CONFIG["results"]["assigned"], show_timer=False)

    # Keep only the last segment for each rid
    last_seg = df.groupby("rid").last().drop(columns=["iti_id", "board_ts"])

    return calculate_egress_time(last_seg)


def calculate_egress_time(df_last_seg: pd.DataFrame) -> pd.DataFrame:
    """
    A helper function designed to calculate the egress time for a specified set of passenger IDs (rids).
    It enriches the input DataFrame with necessary columns and computes the egress time for each passenger.
    The egress time is defined as the time difference between the passenger's exit time and alighting time.

    :param df_last_seg: A DataFrame that holds the last segment information of each itinerary.
                        Each row corresponds to a unique passenger, identified by the "rid" index.

    :return: A DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
             It includes the following columns:
             - "node1": The starting node of the egress path.
             - "node2": The ending node of the egress path.
             - "alight_ts": The time when the passenger alighted from the vehicle.
             - "ts2": The time when the passenger exited the station.
             - "egress_time": The calculated egress time, which is the difference between "ts2" and "alight_ts".

    Raises:
        AssertionError: If the last segment found does not match the last segment in the path,
                        indicating a potential data inconsistency.
    """
    afc = get_afc()
    k_pv = get_k_pv()

    filtered_AFC = afc[np.isin(afc[:, 0], df_last_seg.index)]
    egress_link = k_pv[len(k_pv) - 1 - np.unique(k_pv[:, 0]
                                                 [::-1], return_index=True)[1], :4]
    path_id_node1 = {link[0]: link[2] for link in egress_link}
    path_id_node2 = {link[0]: link[3] for link in egress_link}
    df_last_seg['ts2'] = {record[0]: record[-1] for record in filtered_AFC}
    df_last_seg["UID2"] = {record[0]: record[-2] for record in filtered_AFC}
    df_last_seg["node1"] = df_last_seg["path_id"].map(path_id_node1)
    df_last_seg["node2"] = df_last_seg["path_id"].map(path_id_node2)
    assert df_last_seg[df_last_seg["node2"] != df_last_seg["UID2"]].shape[0] == 0, \
        "Last seg found is not the last seg in path!"
    df_last_seg['egress_time'] = df_last_seg['ts2'] - df_last_seg['alight_ts']
    return df_last_seg[["node1", "node2", "alight_ts", "ts2", "egress_time"]]


def filter_transfer_time_from_assigned() -> pd.DataFrame:
    """
    Find transfer times from assigned_*.pkl files.
    :return: DataFrame with 4 columns:
        ["path_id", "seg_id", "alight_ts", "transfer_time"]
        where seg_id is the alighting train segment id, the transfer time is thus considered as the time difference
        between the alighting time of seg_id and the boarding time of seg_id + 1.
    """
    assigned = read_all(config.CONFIG["results"]["assigned"], show_timer=False)

    # delete (rid, iti_id) with only one seg
    seg_count = assigned.groupby(['rid', 'iti_id'])['seg_id'].transform('nunique')
    assigned = assigned[seg_count > 1]

    # find rows that need to combine next row's board_ts
    assigned['next_index'] = assigned.groupby(['rid', 'iti_id'])['seg_id'].shift(-1).notna()

    # calculate transfer time
    assigned["next_board_ts"] = assigned["board_ts"].shift(-1)
    assigned = assigned[assigned["next_index"]]
    assigned["transfer_time"] = assigned["next_board_ts"] - assigned["alight_ts"]

    # get essential data
    res = assigned[["path_id", "seg_id", "alight_ts", "transfer_time"]]
    return res


def get_transfer_platform_ids_from_path() -> np.ndarray:
    """

    :return: Array with 5 columns:
        ["path_id", "seg_id", "node1", "node2", "transfer_type"]
        where seg_id is the alighting train segment id, the transfer time is thus considered as the time difference
        between the alighting time of seg_id and the boarding time of seg_id + 1.
        where transfer_type is one of "platform_swap", "egress-entry".
    """
    # add seg_id in k_pv
    k_pv_ = get_k_pv()[:, :-2]
    df = pd.DataFrame(k_pv_, columns=["path_id", "pv_id", "node1", "node2", "link_type"])
    in_vehicle_df = df[df['link_type'] == 'in_vehicle'].copy()
    in_vehicle_df['seg_id'] = in_vehicle_df.groupby('path_id').cumcount() + 1
    df = pd.merge(df, in_vehicle_df[['seg_id']], left_index=True, right_index=True, how='left')

    # delete non-transfer paths
    path = read_("path", show_timer=False)
    path_id_with_transfer = path[path["transfer_cnt"] > 0]["path_id"].values
    df = df[df['path_id'].isin(path_id_with_transfer)]

    # delete first and last pathvia row
    first_pv_ind = (df["pv_id"] == 1)  # entry
    last_pv_ind = first_pv_ind.shift(-1)  # egress
    df = df[~(first_pv_ind | last_pv_ind)].iloc[:-1, :]

    # aggregate egress-entry transfer link
    df["seg_id"] = df["seg_id"].ffill().astype(int)
    df["next_node2"] = df["node2"].shift(-1)
    # fix swap expressions
    df.loc[df["link_type"] == "platform_swap", "next_node2"] = df["node2"]
    df.loc[df["link_type"] == "egress", "link_type"] = "egress-entry"  # rename for clarity
    res = df[df["link_type"].isin(["egress-entry", "platform_swap"])][
        ["path_id", "seg_id", "node1", "next_node2", "link_type"]].values
    return res


def map_platform_id_to_platform_uid() -> dict:
    """
    Get a dictionary mapping platform node_ids to platform_uids.

    Platform_uid is the unique identifier of a platform, which can be either nid or smallest
        platform_id on the same physical platform.

    The mapping is based on the platform information and exceptions. (Including Sihe Detour (Huafu Avenue) exception)

    :return: A dictionary with platform node_ids as keys and platform UIDs as values.
    """
    # get platform_id -> nid dict
    node_info = read_(config.CONFIG["results"]["node"], show_timer=False)
    node_info = node_info[node_info["LINE_NID"].notna() & (node_info["IS_TRANSFER"] == 1)]
    p_id2p_uid = {int(k): int(v) for k, v in node_info["STATION_NID"].to_dict().items()}

    # Add Sihe detour (Huafu Avenue) exception: transfer might happens at 10140 with swap
    p_id2p_uid.update({101400: 10140, 101401: 10140, })

    # process platform exceptions
    platform_exceptions = {}  # platform_id -> the smallest platform_id (same physical platform)
    for pl_grps in get_platform().values():
        for pl_grp in pl_grps:
            for platform_id in pl_grp:
                platform_exceptions[platform_id] = min(pl_grp)
    # update platform exceptions
    p_id2p_uid.update(platform_exceptions)

    return p_id2p_uid


def map_path_seg_to_transfer_link() -> pd.DataFrame:
    """
    Map path segments to transfer links.

    :return: A DataFrame with 7 columns:
        ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"]
        where seg_id is the alighting train segment id, transfer_type is one of "platform_swap", "egress-entry",
        and p_uid_min, p_uid_max are the platform_uids of the two platforms involved in the transfer.
    """
    path2trans = get_transfer_platform_ids_from_path()  # path_id, seg_id, node1, node2, transfer_type
    df_p2t = pd.DataFrame(path2trans, columns=["path_id", "seg_id", "node1", "node2", "transfer_type"])

    p_id2p_uid = map_platform_id_to_platform_uid()

    # map platform_id to platform_uid, and then get unique tuple with (p_uid_min, p_uid_max)
    df_p2t["p_uid1"] = df_p2t["node1"].map(p_id2p_uid)
    df_p2t["p_uid2"] = df_p2t["node2"].map(p_id2p_uid)
    df_p2t["p_uid_min"] = df_p2t[["p_uid1", "p_uid2"]].min(axis=1)
    df_p2t["p_uid_max"] = df_p2t[["p_uid1", "p_uid2"]].max(axis=1)
    df_p2t.drop(columns=["p_uid1", "p_uid2"], inplace=True)
    return df_p2t


def get_physical_links_info(et_: pd.DataFrame, platform: dict = None, ) -> np.ndarray:
    """
    Get information about physical links based on platform data and egress times.
    This function generates a structured array containing information about each physical link.
    Each row is an egress link (topological) with the following fields:
        - 'pl_id': A unique identifier for each physical link.
        - 'platform_id': The platform node_id of the egress link.
        - 'uid': The station UID of the egress link.

    :param et_: A DataFrame containing egress times for each rid.
        The DataFrame should have the following columns:
            - 'node2': The station UID of the egress path.
            - 'node1': The platform node_id of the egress path.

    :param platform: A dictionary mapping station UIDs to their corresponding platform node_ids (exceptions).
        The structure of the dictionary is:
                     {
                         'UID': [[node_id_1, node_id_2],...],
                        ...
                     }
                     where `node_id_1`, `node_id_2` are platform nodes.
        Defaults to the result of `read_(fn="platform.json", show_timer=False)`.

    :return: A structured array with the following fields:
        - 'pl_id': A unique identifier for each physical link.
        - 'platform_id': The platform node_id of the egress link.
        - 'uid': The station UID of the egress link.
    """
    platform = platform if platform is not None else read_(
        fn="platform.json", show_timer=False)
    et_ = et_ if et_ is not None else read_(
        fn="egress_times_1.pkl", show_timer=False)

    pl_id = 1
    data = []  # [pl_id, platform_id, uid]
    for uid in range(1001, 1137):
        et = et_[et_["node2"] == uid]
        if et.shape[0] == 0:
            print(uid, "no egress times.")
            continue
        found_platforms = et.node1.unique()
        if uid in platform:
            platform_groups = platform[uid]
            for platform_group in platform_groups:
                platform_group = [
                    p_id for p_id in platform_group if p_id in found_platforms]
                if len(platform_group) == 0:
                    continue
                for platform_id in platform_group:
                    data.append([pl_id, platform_id, uid])
                pl_id += 1
        else:
            nid_dict = {}  # nid -> [platform_id]
            for node1 in found_platforms:
                nid = node1 // 10
                if nid not in nid_dict:
                    nid_dict[nid] = []
                nid_dict[nid].append(node1)

            for nid, platform_ids in nid_dict.items():
                for platform_id in platform_ids:
                    data.append([pl_id, platform_id, uid])
                pl_id += 1
    return np.array(data)


def reject_outlier_bd(data: np.ndarray, method: str = "zscore", abs_max: int | None = 500) -> tuple[float, float]:
    """
    Calculate bounds for outlier rejection.
    see:
        boxplot: https://www.secrss.com/articles/11994
        zscore: https://www.zhihu.com/question/38066650

    :param data: Input data array.
    :param method: Outlier detection method ('zscore' or 'boxplot').
    :param abs_max: Absolute maximum value constraint.
    :return: A tuple of lower and upper bounds for valid data.

    Raises:
        Exception: If an invalid method is provided.
    """
    if method == "boxplot":
        miu = np.mean(data)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        lower_whisker = Q1 - (miu - Q1)  # lower limit
        upper_whisker = Q3 + (Q3 - miu)  # upper limit
        lower_bound, upper_bound = lower_whisker, upper_whisker
    elif method == "zscore":
        lower_bound, upper_bound = np.mean(
            data) - 3 * np.std(data), np.mean(data) + 3 * np.std(data)
    else:
        raise Exception(
            "Please use either boxplot or zscore method to reject outliers!")

    # manual bound
    if abs_max:
        upper_bound = min(upper_bound, abs_max)
    lower_bound = max(lower_bound, 0)

    return lower_bound, upper_bound


def reject_outlier(data: np.ndarray, method: str = "zscore", abs_max: int = 500) -> np.ndarray:
    """
    Reject outliers from the input data array based on the specified method.
    see:
        boxplot: https://www.secrss.com/articles/11994
        zscore: https://www.zhihu.com/question/38066650

    :param data: Input data array.
    :param method: Outlier detection method ('zscore' or 'boxplot').
    :param abs_max: Absolute maximum value constraint.
    :return: cleaned data array.

    Raises:
        Exception: If an invalid method is provided.
    """
    lb, ub = reject_outlier_bd(data, method=method, abs_max=abs_max)
    return data[(data >= lb) & (data <= ub)]


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
        kde_pdf, _ = fit_pdf_cdf(walk_time, method="kde")
        ax_right.plot(kde_pdf(x_values) * scale, x_values,
                      color="red", label="KDE Fit", lw=1)
    if "gamma" in fit_curves:
        # Fit Gamma and LogNormal distributions and plot them on the histogram
        gamma_pdf, _ = fit_pdf_cdf(walk_time, method="gamma")
        ax_right.plot(gamma_pdf(x_values) * scale, x_values,
                      color="green", label="Gamma Fit", lw=1)

    if "lognorm" in fit_curves:
        lognorm_pdf, _ = fit_pdf_cdf(walk_time, method="lognorm")
        ax_right.plot(lognorm_pdf(x_values) * scale, x_values,
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
        "rid" as index. ["node1", "node2", "alight_ts", "ts2", "egress_time"] as columns.

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
        map_p2t: pd.DataFrame,
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

    :param map_p2t: Optional DataFrame mapping physical links to transfer links.

    :param save_subfolder: The subfolder where plots will be saved.
        If not specified, plots are saved in the default figure folder.
    :param save_on: If True, the plots are saved to files. If False, the plots are displayed.
    """
    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print("[INFO] Plotting TTD...")
    df = pd.merge(df_tt, map_p2t, on=["path_id", "seg_id"], how="left")

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


def fit_pdf_cdf(data: np.ndarray, method: str = "kde") -> tuple[Callable, Callable]:
    """
    Fit a probability density function (PDF) and cumulative distribution function (CDF) to the input data.
    Parameters:
        data (np.ndarray): Input data array.
        method (str): Method to fit the PDF and CDF. Options are 'kde' (Kernel Density Estimation),
                      'gamma' (Gamma Distribution), and 'lognorm' (Log-Normal Distribution).
    Returns:
        tuple[Callable, Callable]: A tuple containing the fitted PDF function and CDF function.
                                   The PDF function takes x values as input and returns the corresponding
                                   PDF values. The CDF function takes x values as input and returns the
                                   corresponding CDF values.
    Raises:
        Exception: If an invalid method is provided.
    """
    if method == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        return kde, lambda x_values: np.array([kde.integrate_box_1d(0, x) for x in x_values])
    elif method == "gamma":
        from scipy.stats import gamma
        params = gamma.fit(data, floc=0)
        # print(params)
        return lambda x: gamma.pdf(x, *params), lambda x: gamma.cdf(x, *params)
    elif method == "lognorm":
        from scipy.stats import lognorm
        params = lognorm.fit(data)
        # print(params)
        return lambda x: lognorm(*params).pdf(x), lambda x: lognorm(*params).cdf(x)
    else:
        raise Exception(
            "Please use either kde, gamma, or lognorm method to fit pdf!")


def evaluate_fit(data: np.ndarray, cdf_func: Callable) -> tuple[float, float]:
    """
    Evaluate the fit of a cumulative distribution function (CDF) to the input data.
    :param data: Input data array.
    :param cdf_func: CDF function to evaluate.
    :return: A tuple containing the K-S statistic and the p-value of the K-S test.
    """
    data_sorted = np.sort(data)
    return kstest(data_sorted, cdf_func)


def fit_one_pl(pl_id: int, et_: pd.DataFrame, physical_link_info: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """
    Fit the distribution of physical links egress time with the following columns:
        [
            pl_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]
    :param pl_id: physical link id
    :param et_: egress time dataframe
    :param physical_link_info: physical link info array, each row is [pl_id, platform_id, uid]
    :param x: x values for pdf and cdf, usually [0, 500] with 501 points

    :return: array with shape (x.size, 14) or None
    """
    et = et_[et_["node1"].isin(
        physical_link_info[physical_link_info[:, 0] == pl_id][:, 1])]
    if et.shape[0] == 0:
        return None
    data = et['egress_time'].values
    data = reject_outlier(data, method="zscore", abs_max=500)
    res_this_pl = [np.ones_like(x) * pl_id, x]

    for met in ["kde", "gamma", "lognorm"]:
        pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
        pdf_values = pdf_f(x)
        cdf_values = cdf_f(x)
        cdf_values = cdf_values / cdf_values[-1]  # normalize
        ks_stat, ks_p_val = evaluate_fit(data=data, cdf_func=cdf_f)
        res_this_pl.extend([pdf_values, cdf_values,
                            np.ones_like(x) * ks_stat,
                            np.ones_like(x) * ks_p_val])

    return np.vstack(res_this_pl).T


def fit_egress_time_dis_all_parallel(
        et_: pd.DataFrame,
        physical_link_info: np.ndarray = None,
        n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fit the distribution of physical links egress time with the following columns:
        [
            pl_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]
    :param et_: egress time dataframe
    :param physical_link_info: physical link info array, each row is [pl_id, platform_id, uid]
    :param n_jobs: number of jobs to run in parallel, default is -1 (use all available cores)

    :return: dataframe with shape (n_pl * 501, 14)
    """
    print(
        f"[INFO] Start fitting egress time distribution using {n_jobs} threads...")

    x = np.linspace(0, 500, 501)
    physical_link_info = physical_link_info if physical_link_info is not None else get_physical_links_info(
        et_=et_)
    pl_ids = np.unique(physical_link_info[:, 0])

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_one_pl)(pl_id, et_, physical_link_info, x) for pl_id in
        tqdm(pl_ids, desc="Physical links egress time distribution fitting")
    )
    results = [res for res in results if res is not None]
    res = np.vstack(results)

    return pd.DataFrame(res, columns=[
        "pl_id", "x",
        "kde_pdf", "kde_cdf", "kde_ks_stat", "kde_ks_p_value",
        "gamma_pdf", "gamma_cdf", "gamma_ks_stat", "gamma_ks_p_value",
        "lognorm_pdf", "lognorm_cdf", "lognorm_ks_stat", "lognorm_ks_p_value"
    ])


def fit_transfer_time_dis_all(df_tt: pd.DataFrame, map_p2t: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the distribution of transfer time for all transfer links.

    :param df_tt: Filtered transfer time DataFrame with 4 columns:
        ["path_id", "seg_id", "alight_ts", "transfer_time"]
        where seg_id is the alighting train segment id, transfer_time is the time difference between the alighting
        time of seg_id and the boarding time of seg_id + 1.
    :param map_p2t: DataFrame with 7 columns:
        ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"]
        where seg_id is the alighting train segment id, transfer_type is one of "platform_swap", "egress-entry",
        and p_uid_min, p_uid_max are the platform_uids of the two platforms involved in the transfer.

    :return: DataFrame with 6 columns:
        [p_uid1, p_uid2, x, kde_cdf, gamma_cdf, lognorm_cdf]
        where p_uid1, p_uid2 are platform_uids and p_uid1 is always smaller than p_uid2.
        For platform_swap transfers, only one row with x = 0 and all cdfs = 1.
    """
    print("[INFO] Start fitting transfer time distribution...")
    # merge df_tt and df_p2t to get (p_uid_min, p_uid_max) -> transfer_time info
    df = pd.merge(df_tt, map_p2t, on=["path_id", "seg_id"], how="left")

    x = np.linspace(0, 500, 501)
    res = []
    # for egress-entry transfers:
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "egress-entry"].groupby(["p_uid_min", "p_uid_max"])[
        "transfer_time"]:
        data = df_.values
        print(p_uid1, p_uid2, data.size, end=" -> ")
        data = reject_outlier(data, abs_max=500)
        print(data.size, end="\t | ")

        res_this_transfer = [np.ones_like(x) * p_uid1, np.ones_like(x) * p_uid2, x]
        for met in ["kde", "gamma", "lognorm"]:
            pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
            cdf_values = cdf_f(x)
            cdf_values = cdf_values / cdf_values[-1]  # normalize
            ks_stat, ks_p_val = evaluate_fit(data=data, cdf_func=cdf_f)
            print(f"{met}: {ks_stat:.4f} {ks_p_val:.4f}", end=" | ")
            res_this_transfer.extend([cdf_values])
        res.append(np.vstack(res_this_transfer).T)
        print()
    # for platform_swap transfers: just defaults to 1
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "platform_swap"].groupby(["p_uid_min", "p_uid_max"])[
        "transfer_time"]:
        data = df_.values
        print(p_uid1, p_uid2, data.size)
        res.append(np.array([[
            p_uid1, p_uid2, 0, 1, 1, 1
        ]]))
    res = np.vstack(res)
    res = pd.DataFrame(res, columns=[
        "p_uid1", "p_uid2", "x",
        "kde_cdf", "gamma_cdf", "lognorm_cdf"
    ])
    for col in ["p_uid1", "p_uid2", "x"]:
        res[col] = res[col].astype(int)
    return res


def map_egress_x2pdf(pl_id: int) -> dict[int, float]:
    """
    Get the PDF values for a given physical link id.

    :param pl_id: int
    :return: Dictionary, mapping x to PDF values.
    """
    etd = get_etd()
    etd = etd[etd[:, 0] == pl_id][:, [1, 2]]  # Keep only x and pdf columns
    x2pdf = dict(zip(etd[:, 0], etd[:, 1]))
    return x2pdf


def map_egress_x2cdf(pl_id: int) -> dict[int, float]:
    """
    Get the CDF function for a given physical link id.
    :param pl_id: int
    :return: Dictionary, mapping x to CDF values.
    """
    etd = get_etd()
    etd = etd[etd[:, 0] == pl_id][:, [1, 3]]  # Keep only x and cdf columns
    # Create a dictionary mapping x to cdf
    x2cdf = dict(zip(etd[:, 0], etd[:, 1]))
    return x2cdf


def cal_pdf(x2pdf: dict, walk_time: int | Iterable[int]) -> float | np.ndarray:
    """
    Compute the probability density function (PDF) of walking time.

    :param x2pdf: Dict[int, float], mapping x to PDF values.
        For example, {0: 0.0001, 1: 0.0002, ..., 500: 0.0005}

    :param walk_time: int or Iterable[int]
        Actual walking time(s) to evaluate the PDF. Can be a scalar or a 1D array-like object.

    :return: float or np.ndarray
        PDF value(s) corresponding to the input walking time(s). If walk_time contains values not in range(0, 501),
        the corresponding PDF values will be 0. (indicating zero probability for these values)

    Notes
    -----
    This function is vectorized for performance and can operate efficiently over entire columns.
    """
    # Ensure walk_time is always treated as an array
    walk_time = np.atleast_1d(walk_time)

    # Use NumPy vectorized operations for efficiency
    pdf_values = np.vectorize(x2pdf.get, otypes=[float])(walk_time, 0)

    return pdf_values if pdf_values.size > 1 else pdf_values[0]


def cal_cdf(x2cdf: dict, t_start: int | Iterable[int], t_end: int | Iterable[int]) -> float | np.ndarray:
    """
    Compute the cumulative distribution function (CDF) of walking time between t_start and t_end.

    Parameters
    ----------
    x2cdf : Dict[int, float], mapping x to CDF values.
        For example, {0: 0.0001, 1: 0.0003,..., 500: 1.0000}

    t_start : int or Iterable[int]
        Start time(s) of the interval. Can be a scalar or 1D array-like object.

    t_end : int or Iterable[int]
        End time(s) of the interval. Must be the same shape as `t_start`.

    Returns
    -------
    float or np.ndarray
        CDF value(s) for the interval [t_start, t_end].

    Notes
    -----
    Computes P(walk_time ∈ [t_start, t_end]).
    Efficiently supports vectorized input for use with entire Series/arrays.
    """
    t_start, t_end = np.atleast_1d(t_start), np.atleast_1d(t_end)
    if t_start.shape != t_end.shape:
        raise ValueError("t_start and t_end must have the same shape.")

    cdf_start = np.vectorize(x2cdf.get, otypes=[float])(t_start, None)
    cdf_end = np.vectorize(x2cdf.get, otypes=[float])(t_end, None)
    if np.isnan(cdf_start).any():
        raise ValueError(
            "CDF_start values contain NaN. Please check the t_start ranges.")
    if np.isnan(cdf_end).any():
        raise ValueError(
            "CDF_end values contain NaN. Please check the t_end ranges.")

    return cdf_end - cdf_start if len(cdf_start) > 1 else cdf_end[0] - cdf_start[0]
