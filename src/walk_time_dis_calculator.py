"""
This module contains functions for calculating and mapping various transportation-related data,
specifically focusing on paths, platforms, and transfer links.

Functions:
- `map_path_id_to_platform(egress: bool, entry: bool) -> dict[int, int]`: Maps path IDs to platform IDs,
  distinguishing between egress and entry points.
- `map_platform_id_to_pl_id() -> dict[int, int]`: Maps platform node IDs to physical link IDs.
- `map_path_seg_to_platforms() -> dict[(int, int), (int, int)]`: Maps path segment information to platform IDs
  for transfers.
- `map_pl_id_to_x2pdf_cdf(pdf: bool, cdf: bool) -> dict[int, dict[int, float]]`: Retrieves PDF or CDF values for
  physical links.
- `map_transfer_link_to_x2cdf() -> dict[tuple[int, int], dict[int, float]]`: Retrieves CDF values for transfer
  links.
- `cal_pdf(x2pdf: dict[int, float], walk_time: int | Iterable[int]) -> float | np.ndarray`: Calculates the PDF of
  walking time based on provided data.
- `cal_cdf(x2cdf: dict[int, float], t_start: int | Iterable[int], t_end: int | Iterable[int]) -> float |
  np.ndarray`: Calculates the CDF of walking time between specified start and end times.
"""
from typing import Iterable

import numpy as np
import pandas as pd

from src import config
from src.globals import get_k_pv, get_pl_info, get_etd, get_platform, get_ttd
from src.utils import read_


def map_path_id_to_platform(egress: bool, entry: bool) -> dict[int, int]:
    """
    Get the egress/entry platform id for a given path id.
    :param egress: bool
        If True, return {path_id -> egress platform id}.
    :param entry: bool
        If True, return {path_id -> entry platform id}.
    :return: dict, mapping path_id to platform_id.
    """
    pv = get_k_pv()
    # first row index of each path_id
    first_pv_index = np.where(pv[:, 1] == 1)[0]

    if entry and not egress:
        pv = pv[first_pv_index]
        return {i: j for i, j in pv[:, [0, 3]]}

    if egress:
        pv = pv[first_pv_index - 1]
        return {i: j for i, j in pv[:, [0, 2]]}

    raise Exception("Please specify either egress or entry!")


def map_platform_id_to_pl_id() -> dict[int, int]:
    """
    Convert node_id to physical link id.
    :return: Dictionary mapping platform node_id to physical link id.
    """
    pl_info = get_pl_info()
    res = {i: j for i, j in pl_info[:, [1, 0]]}
    return res


def generate_transfer_info_df_from_path_seg() -> pd.DataFrame:
    """
    Generate a DataFrame containing transfer information from path segment data.

    This method maps pairs of (path_id, seg_id) to information about (p_uid_min, p_uid_max, transfer_type).
    Here, seg_id represents the ID of the alighting train segment, transfer_type can be either "platform_swap"
    or "egress - entry", and p_uid_min and p_uid_max are the unique identifiers of the two platforms involved
    in the transfer.

    Specific steps include:
    1. Obtain the node_ids of transfer platforms from (path_id, seg_id) data.
    2. Establish a mapping from platform node_ids to unique platform identifiers.
    3. Convert platform_ids to unique platform identifiers and calculate the minimum and maximum unique platform
        identifiers for each transfer.

    :return: DataFrame with 7 columns:
            ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"].
    """

    def get_transfer_platform_ids_from_path_seg() -> pd.DataFrame:
        """
        A helper function to get (path_id, seg_id) pair -> (node1, node2, transfer_type)
        :return: DataFrame with 5 columns:
            ["path_id", "seg_id", "node1", "node2", "transfer_type"]
            where seg_id is the alighting train segment id, the transfer time is thus considered as the time difference
            between the alighting time of seg_id and the boarding time of seg_id + 1.
            where transfer_type is one of "platform_swap", "egress-entry".
        """
        # add seg_id in k_pv
        k_pv_ = get_k_pv()[:, :-2]
        df = pd.DataFrame(
            k_pv_, columns=["path_id", "pv_id", "node1", "node2", "link_type"])
        in_vehicle_df = df[df['link_type'] == 'in_vehicle'].copy()
        in_vehicle_df['seg_id'] = in_vehicle_df.groupby(
            'path_id').cumcount() + 1
        df = pd.merge(df, in_vehicle_df[['seg_id']],
                      left_index=True, right_index=True, how='left')

        # delete non-transfer paths
        path = read_("path", show_timer=False)
        path_id_with_transfer = path[path["transfer_cnt"]
                                     > 0]["path_id"].values
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
        df.loc[df["link_type"] == "egress",
               "link_type"] = "egress-entry"  # rename for clarity
        res = df[df["link_type"].isin(["egress-entry", "platform_swap"])][
            ["path_id", "seg_id", "node1", "next_node2", "link_type"]].rename(
            columns={"next_node2": "node2", "link_type": "transfer_type"}
        )
        return res

    def map_platform_id_to_platform_uid() -> dict[int, int]:
        """
        Helper function to get a dictionary mapping platform_ids to platform_uids.

        Platform_uid is the unique identifier of a platform, which can be either station_nid or the
            smallest platform_id on the same physical platform.

        This mapping includes exceptions for the Sihe Detour (Huafu Avenue).

        :return: A dictionary with platform_ids as keys and platform_uids as values.
        """
        # get platform_id -> nid dict
        node_info = read_(config.CONFIG["results"]["node"], show_timer=False)
        node_info = node_info[node_info["LINE_NID"].notna() & (
            node_info["IS_TRANSFER"] == 1)]
        p_id2p_uid = {int(k): int(v)
                      for k, v in node_info["STATION_NID"].to_dict().items()}

        # Add Sihe detour (Huafu Avenue) exception: transfer might happens at 10140 with swap
        p_id2p_uid.update({101400: 10140, 101401: 10140, })

        # process platform exceptions
        # platform_id -> the smallest platform_id (same physical platform)
        platform_exceptions = {}
        for pl_grps in get_platform().values():
            for pl_grp in pl_grps:
                for platform_id in pl_grp:
                    platform_exceptions[platform_id] = min(pl_grp)
        # update platform exceptions
        p_id2p_uid.update(platform_exceptions)

        return p_id2p_uid

    # path_id, seg_id, node1, node2, transfer_type
    df_p2t = get_transfer_platform_ids_from_path_seg()

    p_id2p_uid = map_platform_id_to_platform_uid()

    # map platform_id to platform_uid, and then get unique tuple with (p_uid_min, p_uid_max)
    df_p2t["p_uid1"] = df_p2t["node1"].map(p_id2p_uid)
    df_p2t["p_uid2"] = df_p2t["node2"].map(p_id2p_uid)
    df_p2t["p_uid_min"] = df_p2t[["p_uid1", "p_uid2"]].min(axis=1)
    df_p2t["p_uid_max"] = df_p2t[["p_uid1", "p_uid2"]].max(axis=1)
    df_p2t.drop(columns=["p_uid1", "p_uid2"], inplace=True)
    return df_p2t


def map_path_seg_to_platforms() -> dict[(int, int), (int, int)]:
    """
    Get the transfer platform ids for a given path id.
    :return: dict, mapping (path_id, seg_id) to (platform_uid_min, platform_uid_max).
    """
    df_ps2t = generate_transfer_info_df_from_path_seg(
    )  # path_id, seg_id, node1, node2, p_uid_min, p_uid_max, transfer_type
    data = df_ps2t[["path_id", "seg_id", "p_uid_min", "p_uid_max"]].values
    res = {(p, s): (mi, ma) for p, s, mi, ma in data}
    return res


def map_pl_id_to_x2pdf_cdf(pdf: bool, cdf: bool) -> dict[int, dict[int, float]]:
    """
    Get the PDF / CDF values dictionaries.
    :param pdf: bool
        If True, return {pl_id -> {x -> PDF values}}.
    :param cdf: bool
        If True, return {pl_id -> {x -> CDF values}}.
    :return: dict, mapping pl_id to a dict mapping x to PDF / CDF values.
    """
    etd = get_etd()  # pl_id, x, pdf, cdf
    res = {}
    for pl_id in np.unique(etd[:, 0]):
        if pdf and not cdf:
            x2f = {i: j for i, j in etd[etd[:, 0] == pl_id][:, [1, 2]]}
        elif cdf:
            x2f = {i: j for i, j in etd[etd[:, 0] == pl_id][:, [1, 3]]}
        else:
            raise Exception("Please specify either pdf or cdf!")
        res[pl_id] = x2f
    return res


def map_transfer_link_to_x2cdf() -> dict[tuple[int, int], dict[int, float]]:
    """
    Get the CDF values dictionaries.
    :return: dict, mapping (p_uid_min, p_uid_max) to a dict mapping x to CDF values.
    """
    ttd = get_ttd()  # p_uid_min, p_uid_max, x, cdf
    res = {}
    for p_uid_min, p_uid_max in np.unique(ttd[:, [0, 1]], axis=0):
        x2f = {i: j for i, j in ttd[(ttd[:, 0] == p_uid_min) & (
            ttd[:, 1] == p_uid_max)][:, [2, 3]]}
        res[(p_uid_min, p_uid_max)] = x2f
    return res


def cal_pdf(x2pdf: dict[int, float], walk_time: int | Iterable[int]) -> float | np.ndarray:
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


def cal_cdf(x2cdf: dict[int, float], t_start: int | Iterable[int], t_end: int | Iterable[int]) -> float | np.ndarray:
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
