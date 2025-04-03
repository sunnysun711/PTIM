"""
Please make sure path.pkl and pathvia.pkl are already generated.
"""
import numpy as np

from src.utils import read_data
from src.globals import K_PV_DICT, TT, AFC


def find_trains(nid1: int, nid2: int, ts1: int, ts2: int, line: int, upd: int) -> list[tuple[int, int, int]]:
    """
    Find trains, board_ts, alight_ts for nid pairs within ts range.
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


def find_feas_iti(k_pv: np.ndarray, ts1: int, ts2: int) -> list[list[int | tuple[int, int, int]]]:
    """
    Find feasible itineraries with k shortest path via array (`k_pv`) and ts range (`ts1`,`ts2`) provided.
    :param k_pv: path via array, [path_id, nid1, nid2, line, upd]
    :param ts1: timestamp 1.
    :param ts2: timestamp 2.
    :return: list of itineraries,
        [[path_id, (train_id, b_ts, a_ts), (train_id, b_ts, a_ts), ...], [path_id, (train_id, b_ts, a_ts)], ...]
    """
    feas_iti_list: list[list[int | tuple[int, int, int]]] = []
    for path_id in set(k_pv[:, 0]):
        pv = k_pv[k_pv[:, 0] == path_id]
        # [[(train_id, b_ts, a_ts), (), ...], [], ...], len(seg_trains) = number of seg in this path
        seg_trains: list[list[tuple[int, int, int]]] = []
        ts1_this_seg = ts1
        for seg_id, seg in enumerate(pv):
            trains = find_trains(nid1=seg[1], nid2=seg[2], ts1=ts1_this_seg, ts2=ts2, line=seg[3], upd=seg[4])
            seg_trains.append(trains)
            if not trains:  # cannot find trains, this path_id does not have feasible iti
                break
            ts1_this_seg = min([_[2] for _ in trains])
        else:  # this path_id does have at least one feasible iti
            # do a backward pruning step
            ts2_this_seg = max([_[1] for _ in seg_trains[-1]])
            for trains in reversed(seg_trains[:-1]):
                trains[:] = [t for t in trains if t[2] < ts2_this_seg]  # prune
                ts2_this_seg = max([_[1] for _ in trains])

            # seg_trains ready. Now generate all feasible combinations
            def backtrack(seg_idx: int, current_path: list[tuple[int, int, int]]):
                """Try to build a complete feasible itinerary from seg_trains"""
                if seg_idx == len(seg_trains):
                    feas_iti_list.append([path_id] + current_path)
                    return

                for train in seg_trains[seg_idx]:
                    if not current_path or current_path[-1][2] < train[1]:  # check transfer time > 0
                        current_path.append(train)
                        backtrack(seg_idx + 1, current_path)  # recursion
                        current_path.pop()

            backtrack(0, [])
    return feas_iti_list


if __name__ == '__main__':
    # rid, uid1, ts1, uid2, ts2 = AFC[np.random.choice(len(AFC))]
    rid, uid1, ts1, uid2, ts2 = AFC[AFC[:, 0] == 105057].flatten()
    print(rid, uid1, uid2, ts1, ts2)

    k_pv = K_PV_DICT[(uid1, uid2)]
    print(k_pv)

    print(find_feas_iti(k_pv, ts1, ts2))
    pass
