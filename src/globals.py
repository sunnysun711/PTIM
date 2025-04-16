"""
This module initializes and stores large, frequently used variables
such as the train timetable (TT) and the k-shortest path via dictionary (K_PV_DICT).
These are intended to be imported and read-only across the project.

Key Variables:
1. K_PV_DICT: Mapping from OD station UIDs to k-shortest path segments
2. TT: Preprocessed train timetable matrix
3. AFC: Passenger records with origin-destination-time info
4. K_PV: array of k-shortest paths, full details (including walk links)

Usage:
- Import the module: `import src.globals as gl`
- Access variables: `gl.TT`, `gl.K_PV_DICT`, `gl.AFC`, `gl.K_PV`

Data sources:
- pathvia.pkl
- TT.pkl
- AFC.pkl

Dependencies:
- src.utils: For data reading
"""

import numpy as np

from src import config
from src.utils import read_

K_PV: np.ndarray = np.array([])


def build_k_pv_dic() -> dict[(int, int), np.ndarray]:
    """
    Build a dictionary mapping each (uid1, uid2) station pair to its k-shortest path
    represented by an array of path via segments.

    :return: A dictionary where each key is a (uid1, uid2) tuple representing
             a station OD pair, and each value is an array of shape (k, 5) with columns:
             ["path_id", "nid1", "nid2", "line", "updown"].
    """
    # Read and preprocess path via data
    pv_df = read_(config.CONFIG["results"]["pathvia"], show_timer=False).sort_values(by=["path_id", "pv_id"])
    global K_PV
    K_PV = pv_df.values
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
            np.ndarray: Array of pathvia rows corresponding to this OD pair
        """
        base_path_id = uid1 * 1000000 + uid2 * 100 + 1
        start_idx, end_idx = np.searchsorted(pv_array[:, 0], [base_path_id, base_path_id + 100])
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

    :return: np.ndarray: Array of shape (n, 6) with columns:
        ["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"],
        where `ts1` is the time when the doors open.
    """
    tt_df = read_("TT", show_timer=False).reset_index()
    tt_df = tt_df.sort_values(["LINE_NID", "UPDOWN", "TRAIN_ID", "DEPARTURE_TS"])
    tt_df["ts1"] = tt_df["DEPARTURE_TS"] - tt_df["STOP_TIME"]
    return tt_df[["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"]].values


# ---------------------------
# Public, frequently used variables (read-only across the project)
# ---------------------------
K_PV_DICT = build_k_pv_dic()
TT = build_tt()
AFC = read_("AFC", show_timer=False).drop(columns=["TRAVEL_TIME"]).reset_index().values
