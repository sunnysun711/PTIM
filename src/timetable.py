import numpy as np
import pandas as pd

from src import config
from src.utils import read_all
from src.globals import get_tt


def load_to_std(data: np.ndarray, train_type: str = "a") -> np.ndarray:
    """
    Calculate SPD (Standing Passenger Density) based on in-vehicle passenger number.
    Used a trick to set less-than-0 value to 0 without processing np.NaN values in the array
        def compare_nan_array(func, a, thresh):
            out = ~np.isnan(a)
            out[out] = func(a[out] , thresh)
            return out
    :param data: np.array, passenger count on trains,
    :param train_type: string, "a" or "b"
    :return: np.array
    """
    if train_type not in ["a", "b"]:
        raise TypeError("train_type must be either a or b!")

    train_seats = config.CONFIG["parameters"][f"TRAIN_{train_type.upper()}_SEAT"]
    train_area = config.CONFIG["parameters"][f"TRAIN_{train_type.upper()}_AREA"]

    data = ((data - train_seats) / train_area).astype(float)
    out = ~np.isnan(data)
    out[out] = np.less(data[out], 0)
    data[out] = 0
    return data


def line_to_type(line_nid: int) -> str:
    """
    Convert line_nid to line_type.
    :param line_nid: int
    :return: string, "a" or "b"
    """
    if line_nid in config.CONFIG["parameters"]["A_LINES"]:
        return "a"
    elif line_nid in config.CONFIG["parameters"]["B_LINES"]:
        return "b"
    else:
        raise ValueError("line_nid must be either 1, 2, 3, 4, 7, 10!")


def train_id_to_type(train_id: int) -> str:
    """
    Convert train_id to train_type.
    :param train_id: int
    :return: string, "a" or "b"
    """
    tt = get_tt()[:, [0, 2]]
    tt = tt[tt[:, 0] == train_id]
    line = tt[0, 1]
    return line_to_type(line)


def get_train_load(train_id: int, assigned: pd.DataFrame = None) -> np.ndarray:
    """
    Get train load for a given train_id.
    :param train_id: int
    :param assigned: pd.DataFrame, columns are: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts]
    :return: np.array, columns are: ['sta1_nid', 'dep_ts', 'sta2_nid', 'arr_ts', 'load']
        where 'load' is the number of passengers on the train in the section.
    """
    if assigned is None:
        assigned = read_all(config.CONFIG["results"]["assigned"], show_timer=False)
    assigned = assigned[['train_id', 'board_ts', 'alight_ts']].values
    assigned = assigned[assigned[:, 0] == train_id]

    board_ts, counts = np.unique(assigned[:, 1], return_counts=True)
    board_dic: dict[int, int] = dict(zip(board_ts, counts))  # board_dic.get(key, 0)
    alight_ts, counts = np.unique(assigned[:, 2], return_counts=True)
    alight_dic: dict[int, int] = dict(zip(alight_ts, counts))

    tt = get_tt()
    tt = tt[tt[:, 0] == train_id][:, [1, 4, 5]]

    this_tt = np.hstack([tt[:-1, [0, 2]], tt[1:, [0, 1]]])  # sta1, dep_ts, sta2, arr_ts
    on_train_psg_num = 0
    sec_psg = []
    for sta1, arr_ts, dep_ts in tt:
        alight_num = alight_dic.get(arr_ts, 0)
        board_num = board_dic.get(dep_ts, 0)
        on_train_psg_num += board_num - alight_num
        sec_psg.append(on_train_psg_num)
    if sec_psg[-1] != 0:
        print(f"\033[31m[ERROR] Train {train_id} is not empty at the end of the trip!\033[0m")
        raise ValueError("Num of passengers boarding and alighting should be equal!")
    sec_psg = sec_psg[:-1]
    this_tt = np.hstack([this_tt, np.array(sec_psg).reshape(-1, 1)])
    return this_tt


if __name__ == '__main__':
    config.load_config()
    train_id = np.random.choice(get_tt()[:, 0], size=1)
    train_type = train_id_to_type(train_id)
    print(train_id, train_type)
    loaded_tt = get_train_load(train_id)
    print(loaded_tt)
    print(load_to_std(loaded_tt[:, -1], train_type))
