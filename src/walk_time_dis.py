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

from src.globals import AFC, K_PV
from src.utils import read_data, read_data_all


def get_egress_time_from_feas_iti_left() -> np.ndarray:
    """
    Find rids in feas_iti_left.pkl where all feasible itineraries share the same final train_id.

    :return: NumPy array with shape (n_rids, 3) containing
        "node1", "node2", "alight_ts", "ts2", "egress_time".
    """
    df = read_data("feas_iti_left", show_timer=False)

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


def get_egress_time_from_feas_iti_assigned() -> np.ndarray:
    """
    Find rids in all feas_iti_assigned_*.pkl files where all feasible itineraries share the same final train_id.

    :return: NumPy array with shape (n_rids, 5) containing
        ["node1", "node2", "alight_ts", "ts2", "egress_time"].
    """
    df = read_data_all("feas_iti_assigned", show_timer=False)

    # Keep only the last segment for each rid
    last_seg = df.groupby("rid").last().drop(columns=["iti_id", "board_ts"])

    return calculate_egress_time(last_seg)


def calculate_egress_time(df_last_seg: pd.DataFrame) -> np.ndarray:
    """
    Helper function to calculate egress time for a given set of rids. This function adds the necessary columns
    to the DataFrame and calculates the egress time for each passenger.

    :param df_last_seg: DataFrame containing the last segment of each itinerary.
    :return: NumPy array with shape (n_rids, 5) containing
        ["node1", "node2", "alight_ts", "ts2", "egress_time"].
    """
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
    res = df_last_seg[["node1", "node2", "alight_ts", "ts2", "egress_time"]].values
    return res


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
