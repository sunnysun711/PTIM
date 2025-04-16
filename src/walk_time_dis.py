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
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns

from src import config
from src.utils import read_, read_all, ts2tstr


def get_egress_time_from_feas_iti_left() -> pd.DataFrame:
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


def get_egress_time_from_feas_iti_assigned() -> pd.DataFrame:
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
    from src.globals import AFC, K_PV
    filtered_AFC = AFC[np.isin(AFC[:, 0], df_last_seg.index)]
    egress_link = K_PV[len(K_PV) - 1 - np.unique(K_PV[:, 0][::-1], return_index=True)[1], :4]
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


def get_egress_link_groups(
        platform: dict = None,
        et_: pd.DataFrame = None,
) -> dict[int, list[list[tuple[int, int]]]]:
    """
    Generate egress link groups based on platform data and egress times.
    This function groups the egress links for each station UID based on available platform data
    and egress times for each rid. It returns a dictionary mapping each station UID to a list of
    egress link groups. Each link group contains pairs of nodes that share same physical platforms.

    :param platform: A dictionary mapping station UIDs to their corresponding platform node_ids (exceptions).
        The structure of the dictionary is:
                     {
                         'UID': [[node_id_1, node_id_2], ...],
                         ...
                     }
                     where `node_id_1`, `node_id_2` are platform nodes.
        Defaults to the result of `read_(fn="platform.json", show_timer=False)`.
    :param et_: A DataFrame containing egress times for each rid.
        The DataFrame should have the following columns:
            - 'node2': The station UID of the egress path.
            - 'node1': The platform node_id of the egress path.
        Defaults to the result of `read_(fn=f"egress_times_1.pkl", show_timer=False)`.
    :return: A dictionary mapping each station UID to a list of egress link groups.
        The structure of the dictionary is:
                     {
                         'UID': [[(node_id_1, UID), (node_id_2, UID)],...],
                        ...
                     }
        Example:
                     {
                         1031: [[(102241, 1031), (102240, 1031)]],
                         1032: [[(104290, 1032), (102320, 1032)], [(104291, 1032), (102321, 1032)]],
                         ...
                     }

    Notes:
        - Each link group is a list of tuples, where each tuple represents a link from node1 to uid.
        - Links within the same list should share same physical platforms.
        - If a station UID is not found in the egress times DataFrame, a message is printed and the UID is skipped.

    """
    platform = platform if platform is not None else read_(fn="platform.json", show_timer=False)
    et_ = et_ if et_ is not None else read_(fn="egress_times_1.pkl", show_timer=False)
    uid2linkgrp = {}
    for uid in range(1001, 1137):
        et = et_[et_["node2"] == uid]
        if et.shape[0] == 0:
            print(uid, "no egress times.")
            continue

        found_platforms = et.node1.unique()
        if str(uid) in platform:
            platform_values = platform[str(uid)]
            link_grps = []
            for sub_list in platform_values:
                new_sub_list = [(id, uid) for id in sub_list if id in found_platforms]
                if new_sub_list:
                    link_grps.append(new_sub_list)
        else:
            nid_dict = {}
            for node1 in found_platforms:
                nid = node1 // 10
                if nid not in nid_dict:
                    nid_dict[nid] = []
                nid_dict[nid].append((int(node1), uid))

            link_grps = list(nid_dict.values())
        uid2linkgrp[uid] = link_grps
    return uid2linkgrp


def get_reject_outlier_bd(data: np.ndarray, method: str = "zscore", abs_max: int = None) -> tuple[float, float]:
    """
    reject outliers of data using zscore or boxplot method
    see:
        boxplot: https://www.secrss.com/articles/11994
        zscore: https://www.zhihu.com/question/38066650

    :param data:
    :param method: using what method to delete outliers.
    :param abs_max: absolute max value of data. If data > abs_max, delete it.
    :return:
    """
    # Initialize bounds
    lower_bound, upper_bound = 0, 500

    # First reject outliers by method
    if method == "boxplot":
        miu = np.mean(data)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        lower_whisker = Q1 - (miu - Q1)  # lower limit
        upper_whisker = Q3 + (Q3 - miu)  # upper limit
        lower_bound, upper_bound = lower_whisker, upper_whisker
    elif method == "zscore":
        lower_bound, upper_bound = np.mean(data) - 3 * np.std(data), np.mean(data) + 3 * np.std(data)
    else:
        raise Exception("Please use either boxplot or zscore method to reject outliers!")

    # Then reject outliers manually
    if abs_max:
        upper_bound = min(upper_bound, abs_max)

    return lower_bound, upper_bound


def plot_egress_time_dis(egress_time: np.ndarray, alight_ts: np.ndarray, title: str = "", show_: bool = True):
    # Set up the figure and axes for the grid layout
    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 0.25])

    # Histogram of the egress time distribution (all day)
    ax_hist_right = fig.add_subplot(grid[0, 1])
    sns.histplot(y=egress_time, kde=True, ax=ax_hist_right, color="blue", bins=30)
    ax_hist_right.set_xlabel("Frequency")
    ax_hist_right.set_ylabel("")

    # Scatter plot in the center
    ax_scatter = fig.add_subplot(grid[0, 0], sharey=ax_hist_right)
    ax_scatter.scatter(alight_ts, egress_time, alpha=0.4)
    # ax_scatter.set_xlabel("Alight Timestamp")
    ax_scatter.set_ylabel("Egress Time")
    ax_scatter.set_xticks(range(6 * 3600, 24 * 3600 + 1, 3600))
    ax_scatter.set_xticklabels([f"{i:02}:00" for i in range(6, 25, 1)])
    ax_scatter.set_xlim(6 * 3600, 24 * 3600)
    ax_scatter.set_title("Egress Time Distribution " + title)

    # Boxplot of egress time versus alight_ts, with customized bin width
    _bin_width = 1800
    alight_ts_binned = (alight_ts // _bin_width) * _bin_width

    ax_box = fig.add_subplot(grid[1, 0])
    sns.boxplot(x=alight_ts_binned, y=egress_time, ax=ax_box)
    ax_box.set_xlabel(f"Alight Timestamp")
    ax_box.set_ylabel("Egress Time")
    ax_box.set_xticks([i - 0.5 for i in range((24 - 6) * 3600 // _bin_width + 1)])
    ax_box.set_xticklabels([ts2tstr(ts) for ts in range(6 * 3600, 24 * 3600 + 1, _bin_width)])
    ax_box.set_xlim(- 0.5, (24 - 6) * 3600 // _bin_width - 0.5)
    for label in ax_box.get_xticklabels():
        label.set_rotation(90)

    plt.tight_layout()
    if show_:  # Show the plot
        plt.show()
    else:  # Save the plot
        fig.savefig(fname=f"figures/egress_time_dis_{title}.pdf", dpi=600)
        plt.clf()
    return


def plot_egress_time_dis_all():
    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False)
    et__ = et_.set_index(["node1", "node2"])
    for uid, platform_links in get_egress_link_groups(et_=et_).items():
        print(uid, platform_links)
        for egress_links in platform_links:
            # all egress links on the same platform
            et = et__[et__.index.isin(egress_links)]
            if et.shape[0] == 0:
                continue
            print("Data size: ", et.shape[0], egress_links)
            lb, ub = get_reject_outlier_bd(data=et['egress_time'].values, method="zscore", abs_max=500)
            et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
            print(f"Bounded by [{lb}, {ub}]: ", et.shape[0])

            plot_egress_time_dis(
                egress_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=f"{egress_links}",
                show_=False  # save
            )
    return


def fit_walk_time_distribution():
    ...


def get_pdf(walk_param: dict[str, float], walk_time: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the probability density function (PDF) of walking time.

    Parameters
    ----------
    walk_param : dict[str, float]
        Dictionary containing parameters of the walking time distribution.
        Must include keys: 'mu', 'sigma' (for log-normal distribution).

    walk_time : float or Iterable[float]
        Actual walking time(s) to evaluate the PDF. Can be a scalar or a 1D array-like object.

    Returns
    -------
    float or np.ndarray
        PDF value(s) corresponding to the input walking time(s).

    Notes
    -----
    This function is vectorized for performance and can operate efficiently over entire columns.
    """
    ...


def get_cdf(walk_param: dict[str, float], t_start: float | np.ndarray, t_end: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the cumulative distribution function (CDF) of walking time between t_start and t_end.

    Parameters
    ----------
    walk_param : dict[str, float]
        Dictionary containing parameters of the walking time distribution.
        Must include keys: 'mu', 'sigma' (for log-normal distribution).

    t_start : float or Iterable[float]
        Start time(s) of the interval. Can be a scalar or 1D array-like object.

    t_end : float or Iterable[float]
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
    ...


if __name__ == '__main__':
    # Load the configuration using the config file path
    config.load_config(config_file="configs/config1.yaml")
