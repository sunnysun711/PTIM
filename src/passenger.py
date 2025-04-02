"""
Please make sure path.pkl and pathvia.pkl are already generated.
"""
import numpy as np
from src.utils import read_data

# get frequently used data: pathvia array
PV = read_data("pathvia.pkl", show_timer=False).sort_values(["path_id"])
# PV = read_data("pathvia.pkl", show_timer=False).sort_values(by=["path_id", "pv_id"])
PV: np.ndarray = PV[PV["link_type"] == "in_vehicle"][["path_id", "node_id1", "node_id2", "line", "updown"]].values

# get frequently used data: timetable array
TT = read_data("TT", show_timer=False).reset_index().sort_values(["LINE_NID", "UPDOWN", "TRAIN_ID", "DEPARTURE_TS"])
TT['ts1'] = TT["DEPARTURE_TS"] - TT["STOP_TIME"]  # make sure the carriage gate is open at ts1
TT: np.ndarray[int] = TT[["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ts1", "DEPARTURE_TS"]].values


def find_k_pv(_uid1: int, _uid2: int) -> np.ndarray:
    """
    Find pathvia array for k shortest paths.

    :return: [path_id, node1, node2, line, updown]
    """
    base_path_id = _uid1 * 1000000 + _uid2 * 100 + 1
    start_idx, end_idx = np.searchsorted(PV[:, 0], [base_path_id, base_path_id + 100])
    return PV[start_idx:end_idx]


def find_tt(nid1: int, nid2: int, ts1: int, ts2: int, line:int, upd: int) -> list[tuple[int, int, int]]:
    """
    Find timetable array for nid pairs within ts range.
    Board_ts is obtained from departure_ts, alight_ts from arrive_ts.

    :return: [(train_id, board_ts, alight_ts), ...]
    """
    assert ts1 < ts2, f"ts1 should be smaller than ts2: {ts1}, {ts2}"

    # filter line
    start_idx, end_idx = np.searchsorted(TT[:, 2], [line, line + 1])
    tt: np.ndarray[int] = TT[start_idx:end_idx]

    # filter updown
    start_idx, end_idx = np.searchsorted(tt[:, 3], [upd, upd + 1])
    tt = tt[start_idx:end_idx]

    # filter od and ts range
    tt = tt[
        ((tt[:, 1] == nid1) | (tt[:, 1] == nid2))  # filter nid
        & (tt[:, 5] > ts1)  # filter ts1
        & (tt[:, 4] < ts2)  # filter ts2
        ]

    # Pair finding using vectorized comparison
    train_ids = tt[:-1, 0] == tt[1:, 0]  # check if train_id is the same for consecutive rows
    nid1_condition = tt[:-1, 1] == nid1  # check if the first station is _nid1
    nid2_condition = tt[1:, 1] == nid2  # check if the second station is _nid2

    # Combine conditions
    condition = train_ids & nid1_condition & nid2_condition

    # Prepare result using filtered indices
    trains = [(tt[i, 0], tt[i, 5], tt[i + 1, 4]) for i in range(len(tt) - 1) if condition[i]]
    return trains





if __name__ == '__main__':
    pass
