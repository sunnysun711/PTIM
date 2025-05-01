import numpy as np
import pandas as pd

from src import config
from src.utils import read_all
from src.globals import get_tt

TI2C: dict[int, tuple[float, float]] = None


def get_ti2c() -> dict[int, tuple[float, float]]:
    """
    Get train_id to capacity mapper. Including capacities of normal passengers and commuters.
    
    Returns:
        dict[int, tuple[float, float]]: train_id to capacity mapper.
        {train_id: (cap, cap_max)}
    """
    global TI2C
    if TI2C is None:
        tt = get_tt()[:, :3]  # ["TRAIN_ID", "STATION_NID", "LINE_NID", ... ]
        train_ids = np.unique(tt[:, 0])
        typeA_cap = config.CONFIG["parameters"]["TRAIN_A_AREA"] * \
            config.CONFIG["parameters"]["STD_NORMAL"] + \
            config.CONFIG["parameters"]["TRAIN_A_SEAT"]
        typeA_cap_max = config.CONFIG["parameters"]["TRAIN_A_AREA"] * \
            config.CONFIG["parameters"]["STD_COMMUTER"] + \
            config.CONFIG["parameters"]["TRAIN_A_SEAT"]
        typeB_cap = config.CONFIG["parameters"]["TRAIN_B_AREA"] * \
            config.CONFIG["parameters"]["STD_NORMAL"] + \
            config.CONFIG["parameters"]["TRAIN_B_SEAT"]
        typeB_cap_max = config.CONFIG["parameters"]["TRAIN_B_AREA"] * \
            config.CONFIG["parameters"]["STD_COMMUTER"] + \
            config.CONFIG["parameters"]["TRAIN_B_SEAT"]

        TI2C = {}
        for train_id in train_ids:
            line_nid = tt[tt[:, 0] == train_id][0, 2]
            if line_nid in config.CONFIG["parameters"]["A_LINES"]:
                TI2C[train_id] = (typeA_cap, typeA_cap_max)
            elif line_nid in config.CONFIG["parameters"]["B_LINES"]:
                TI2C[train_id] = (typeB_cap, typeB_cap_max)
            else:
                raise ValueError(f"line_nid {line_nid} not supported!")
    return TI2C


def convert_load_to_density(data: np.ndarray, train_type: str = "a") -> np.ndarray:
    """
    Calculate SPD (Standing Passenger Density) based on in-vehicle passenger number.
    Used a trick to set less-than-0 value to 0 without processing np.NaN values in the array
        def compare_nan_array(func, a, thresh):
            out = ~np.isnan(a)
            out[out] = func(a[out] , thresh)
            return out
    Args:
        data (np.ndarray): passenger count on trains,
        train_type (str, optional): string, "a" or "b". Defaults to "a".

    Returns:
        np.ndarray: SPD of the train, same shape as `data`.
    
    Example:
        >>> data = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        >>> train_type = "a"
        >>> convert_load_to_density(data, train_type)
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


def calculate_train_load_profile(train_id: int, board_records: np.ndarray = None) -> np.ndarray:
    """
    Calculate train load for a given train_id.
    
    Args:
        train_id (int): train_id
        board_records (np.ndarray): boarded records from assigned pkl files,
            columns are: [board_ts, alight_ts].
            Default is None, which will read and process from all assigned_*.pkl files.
    
    Returns:
        np.ndarray: columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
            where 'load' is the number of passengers on the train in the section.
    """
    if board_records is None:
        assigned = read_all(
            config.CONFIG["results"]["assigned"], show_timer=False)
        board_records = assigned[assigned["train_id"] ==
                                train_id][["board_ts", "alight_ts"]].values

    board_ts, counts = np.unique(board_records[:, 0], return_counts=True)
    board_dic: dict[int, int] = dict(
        zip(board_ts, counts))  # board_dic.get(key, 0)
    alight_ts, counts = np.unique(board_records[:, 1], return_counts=True)
    alight_dic: dict[int, int] = dict(zip(alight_ts, counts))

    tt = get_tt()
    tt = tt[tt[:, 0] == train_id][:, [1, 4, 5]]

    this_tt = np.hstack([tt[:-1, [0, 2]], tt[1:, [0, 1]]]
                        )  # sta1, dep_ts, sta2, arr_ts
    on_train_psg_num = 0
    sec_psg = []
    for sta1, arr_ts, dep_ts in tt:
        alight_num = alight_dic.get(arr_ts, 0)
        board_num = board_dic.get(dep_ts, 0)
        on_train_psg_num += board_num - alight_num
        sec_psg.append(on_train_psg_num)
    if sec_psg[-1] != 0:
        print(
            f"\033[31m[ERROR] Train {train_id} is not empty at the end of the trip!\033[0m")
        raise ValueError(
            "Num of passengers boarding and alighting should be equal!")
    sec_psg = sec_psg[:-1]
    this_tt = np.hstack([this_tt, np.array(sec_psg).reshape(-1, 1)])
    return this_tt


def find_overload_train_section(assigned: pd.DataFrame = None) -> dict[int, np.ndarray]:
    """
    Find the overload train, section pairs.

    Args:
        assigned (pd.DataFrame, optional): assigned dataframe. Defaults to None.
            columns are: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts]

    Returns:
        dict[int, np.ndarray]: Dictionary mapping train_id to np.ndarray, 
            columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
            where 'load' is the number of passengers on the train in the section.
            Only overloaded train_id will be included in the dictionary.
            Only overloaded sections will be included in the np.ndarray.
    """
    if assigned is None:
        assigned = read_all(
            config.CONFIG["results"]["assigned"], show_timer=False)

    res: dict[int, np.ndarray] = {}
    for train_id, board_counts in assigned["train_id"].value_counts().items():
        cap, cap_max = get_ti2c()[train_id]
        # to test this function, override cap and cap_max with manual values
        # cap, cap_max = 500, 800

        if board_counts < cap:  # all boarded less than normal capacity, skip
            continue

        # ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
        load_arr = calculate_train_load_profile(
            train_id, assigned[assigned["train_id"] == train_id])
        if np.all(load_arr[:, -1] < cap):  # section passengers less than capacity, skip
            continue

        if np.any(load_arr[:, -1] > cap_max):
            print(
                f"\033[33m[WARNING] Train {train_id} exceeds max capacity!\033[0m")

        res[train_id] = load_arr[load_arr[:, -1] > cap]
    
    return res


if __name__ == '__main__':
    config.load_config()
    train_id = np.random.choice(get_tt()[:, 0], size=1)[0]
    line_nid = get_tt()[get_tt()[:, 0] == train_id][0, 2]
    train_type = "a" if line_nid in config.CONFIG["parameters"]["A_LINES"] else "b"
    print(train_id, train_type)
    loaded_tt = calculate_train_load_profile(train_id)
    print(loaded_tt)
    print(convert_load_to_density(loaded_tt[:, -1], train_type))
