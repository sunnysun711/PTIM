"""
This module provides global variables and utility functions for accessing
preprocessed data used throughout the project. These variables are initialized
on-demand and cached for efficient reuse.

Key Variables:
1. K_PV_DICT: Dictionary mapping station OD pairs to k-shortest path segments.
2. TT: Train timetable matrix with departure and arrival times.
3. AFC: Passenger records with origin-destination and timestamp information.
4. PL_INFO: Physical link information mapping platforms to station UIDs.
5. ETD: Egress time distribution data for physical links.

Usage:
- Import the module: `import src.globals as gl`
- Access variables: `gl.get_tt()`, `gl.get_k_pv_dict()`, etc.

Data Sources:
- pathvia.pkl
- TT.pkl
- AFC.pkl
- physical_links.csv
- etd.pkl

Dependencies:
- src.utils: For data reading and preprocessing.
"""

import numpy as np
import pandas as pd

from src import config
from src.utils import read_

K_PV: np.ndarray | None = None
K_PV_DICT: dict[(int, int), np.ndarray] | None = None
TT: np.ndarray | None = None
AFC: np.ndarray | None = None
PL_INFO: np.ndarray | None = None
ETD: np.ndarray | None = None
LINK_INFO: np.ndarray | None = None
PLATFORM: dict | None = None


def build_k_pv_dic() -> dict[(int, int), np.ndarray]:
    """
    Build a dictionary mapping each (uid1, uid2) station pair to its k-shortest path
    represented by an array of path via segments.

    :return: A dictionary where each key is a (uid1, uid2) tuple representing
             a station OD pair, and each value is an array of shape (k, 5) with columns:
             ["path_id", "nid1", "nid2", "line", "updown"].
    """
    # Read and preprocess path via data
    pv_df = read_(config.CONFIG["results"]["pathvia"], show_timer=False).sort_values(
        by=["path_id", "pv_id"])
    pv_df["nid1"] = pv_df["node_id1"] // 10
    pv_df["nid2"] = pv_df["node_id2"] // 10

    # Filter only in-vehicle segments and extract relevant columns
    pv_array: np.ndarray = pv_df[pv_df["link_type"] == "in_vehicle"][
        ["path_id", "nid1", "nid2", "line", "updown"]
    ].values

    def _find_k_pv(uid1: int, uid2: int) -> np.ndarray:
        """
        Retrieve all pathvia segments for a given (uid1, uid2) pair.

        Args:
            uid1 (int): Origin station UID
            uid2 (int): Destination station UID

        Returns:
            Array of pathvia rows corresponding to this OD pair
        """
        base_path_id = uid1 * 1000000 + uid2 * 100 + 1
        start_idx, end_idx = np.searchsorted(
            pv_array[:, 0], [base_path_id, base_path_id + 100])
        return pv_array[start_idx:end_idx]

    return {
        (_uid1, _uid2): _find_k_pv(_uid1, _uid2)
        for _uid1 in range(1001, 1137)
        for _uid2 in range(1001, 1137)
        if _uid1 != _uid2
    }


def build_tt() -> np.ndarray[int]:
    """
    Load and preprocess the train timetable data.

    :return: Array of shape (n, 6) with columns:
        ["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"],
        where `ts1` is the time when the doors open.
    """
    tt_df = read_("TT", show_timer=False).reset_index()
    tt_df = tt_df.sort_values(
        ["LINE_NID", "UPDOWN", "TRAIN_ID", "DEPARTURE_TS"])
    tt_df["ts1"] = tt_df["DEPARTURE_TS"] - tt_df["STOP_TIME"]
    return tt_df[["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"]].values


def get_k_pv() -> np.ndarray:
    """
    Get global variable K_PV.
    If K_PV is not initialized, read from pathvia.pkl and store in K_PV.

    :return: Array of k-shortest paths, full details (including walk links)
        with columns: ["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown"].
    """
    global K_PV
    if K_PV is None:
        K_PV = read_(config.CONFIG["results"]["pathvia"], show_timer=False).sort_values(
            by=["path_id", "pv_id"]).values
    return K_PV


def get_k_pv_dict() -> dict[(int, int), np.ndarray]:
    """
    Get global variable K_PV_DICT.
    If K_PV_DICT is not initialized, build it using build_k_pv_dic().

    :return: A dictionary where each key is a (uid1, uid2) tuple representing
             a station OD pair, and each value is an array of shape (k, 5) with columns:
             ["path_id", "nid1", "nid2", "line", "updown"].
    """
    global K_PV_DICT
    if K_PV_DICT is None:
        K_PV_DICT = build_k_pv_dic()
    return K_PV_DICT


def get_tt() -> np.ndarray:
    """
    Get global variable TT.
    If TT is not initialized, build it using build_tt().

    :return: Array of shape (n, 6) with columns:
        ["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"],
        where `ts1` is the time when the doors open.
    """
    global TT
    if TT is None:
        TT = build_tt()
    return TT


def get_afc() -> np.ndarray:
    """
    Get global variable AFC.
    If AFC is not initialized, read from AFC.pkl and store in AFC.

    :return: Array of shape (n, 5) with columns:
        [RID, STATION_UID1, TS1, STATION_UID2, TS2],
        where `TS1` and `TS2` are the tap-in and tap-out times.
    """
    global AFC
    if AFC is None:
        AFC = read_("AFC", show_timer=False).drop(
            columns=["TRAVEL_TIME"]).reset_index().values
    return AFC


def get_pl_info() -> np.ndarray:
    """
    Get global variable PL_INFO.
    If PL_INFO is not initialized, read from the latest version of physical_links.csv and store in PL_INFO.
    :return: Array of shape (n, 3) with columns:
        ["pl_id", "platform_id", "uid"].
        where `pl_id` is the physical link ID, `platform_id` is the platform node_id, and `uid` is the station UID.
    """
    global PL_INFO
    if PL_INFO is None:
        PL_INFO = read_(
            config.CONFIG["results"]["physical_links"], show_timer=False, latest_=True).values
    return PL_INFO


def get_etd() -> np.ndarray:
    """
    Get global variable ETD.
    If ETD is not initialized, read from the latest version of etd.pkl and store in ETD.
    :return: Array of shape (n, 4) with columns:
        ['pl_id', 'x', 'pdf', 'cdf']
        where `pl_id` is the physical link ID, `x` is egress time index (0-500).
    """
    global ETD
    if ETD is None:
        ETD = read_(config.CONFIG["results"]["etd"],
                    show_timer=False, latest_=True)
        ETD = ETD[[
            "pl_id", "x", f"{config.CONFIG['parameters']['distribution_type']}_pdf",
            f"{config.CONFIG['parameters']['distribution_type']}_cdf",
            # f"{config.CONFIG['parameters']['distribution_type']}_ks_stat",
            # f"{config.CONFIG['parameters']['distribution_type']}_ks_p_value"
        ]]
        ETD.columns = [
            "pl_id", "x", "pdf", "cdf",
            # "ks_stat", "ks_p_value"
        ]
    return ETD.values


def get_link_info() -> np.ndarray:
    """
    Get global variable LINK_INFO.
    If LINK_INFO is not initialized, read from link_info.pkl and store in LINK_INFO.
    :return: Array of shape (n, 4) with columns:
        ["weight", "node_id1", "node_id2", "link_type"].
    """
    global LINK_INFO
    if LINK_INFO is None:
        LINK_INFO = read_(
            config.CONFIG["results"]["link"], show_timer=False, latest_=False).values
    return LINK_INFO


def get_platform() -> dict:
    """
    Get global variable PLATFORM.
    If PLATFORM is not initialized, read from platform.json and store in PLATFORM.
    :return: Dictionary with platform information.
    """
    global PLATFORM
    if PLATFORM is None:
        PLATFORM = read_("platform.json", show_timer=False, latest_=False)
    return PLATFORM
