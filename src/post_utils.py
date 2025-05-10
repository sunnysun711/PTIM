"""

Usage:
```python
# prepare traj
traj = read_all(...)

# add necessary columns to traj
traj = add_cols_to_traj(traj)
traj = add_platform_arrival_ts_to_traj(traj)

# count
times, psg = count_in_station_psg(traj, node_id=123)
# or
times, psg = count_platform_waiting_psg_with_distribution(traj, node_id=123)
```
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.globals import get_afc, get_etd, get_k_pv, get_platform, get_tt
from src.utils import read_, read_all
from src.walk_time_dis_calculator import WalkTimeDisModel


def add_seg_id_to_k_pv(k_pv: np.ndarray) -> np.ndarray:
    """
    add seg_id columns to k_pv numpy array.
    k_pv is a numpy array with shape (*, ).

    return cols: [`path_id, pv_id, node_id1, node_id2, link_type, line, updown, seg_id`]

    :param k_pv: k path via array. 
        [path_id, pv_id, node_id1, node_id2, link_type, line, updown]
    :type k_pv: np.ndarray

    :return: k_pv with added column: seg_id
    :rtype: np.ndarray
    """
    # add seg_id column
    k_pv = np.hstack((k_pv, np.zeros((k_pv.shape[0], 1))))

    seg_id = 1
    cur_path_id = k_pv[0, 0]

    for i in range(k_pv.shape[0]):
        if k_pv[i, -2] == 0:  # skip walk links
            continue
        if k_pv[i, 0] == cur_path_id:
            k_pv[i, -1] = seg_id
            seg_id += 1
        else:
            cur_path_id = k_pv[i, 0]
            seg_id = 1
            k_pv[i, -1] = seg_id
            seg_id += 1
    return k_pv
    ...


def add_cols_to_traj(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Add [node_id1, node_id2, line, updown, prev_ts] columns to trajectory dataframe.

    :param traj: trajectory dataframe.
    :type traj: pd.DataFrame
    :return: dataframe with added columns.
    :rtype: pd.DataFrame
    """
    # remove columns avoid duplicate names
    for col in ["node_id1", "node_id2", "line", "updown"]:
        if col in traj.columns:
            traj.drop(columns=[col], inplace=True)
    
    afc = get_afc()
    k_pv = add_seg_id_to_k_pv(get_k_pv())

    df_k_pv = pd.DataFrame(k_pv, columns=[
        "path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown", "seg_id"
    ])
    traj = traj.merge(df_k_pv[["path_id", "seg_id", "node_id1", "node_id2", "line", "updown"]],
                      on=["path_id", "seg_id"], how='left')
    traj["prev_ts"] = traj.groupby("rid")["alight_ts"].shift(1)

    df_rid_ts1 = pd.DataFrame(afc[:, [0, 2]], columns=["rid", "ts1"])
    traj = traj.merge(df_rid_ts1, on="rid", how="left")
    traj.loc[traj["seg_id"] == 1,
             "prev_ts"] = traj.loc[traj["seg_id"] == 1, "ts1"]
    traj.drop(columns=["ts1"], inplace=True)
    return traj


def count_left_behind_times(traj: pd.DataFrame, platform_arrival_ts_col:str) -> pd.DataFrame:
    """
    add left-behind-times columns to trajectory dataframe.
    
    input traj must have already added columns: [node_id1, node_id2, `platform_arrival_ts_col`]

    :param traj: trajectory dataframe with node_id1, node_id2, `platform_arrival_ts_col` columns.
    :type traj: pd.DataFrame
    :param platform_arrival_ts_col: the column name to be used for counting left-behind-times.
    :type platform_arrival_ts_col: str, should be 'prev_ts' or 'platform_arrival_ts'
    :return: dataframe with added columns: [left_behind_count]
    :rtype: pd.DataFrame
    """
    tt = get_tt()
    dep_times_per_node = {
        node: np.sort(tt[(tt[:, 1] == node // 10) &
                      (tt[:, 3] == (1 if node % 10 == 0 else -1))
        ][:, 5])
        for node in traj["node_id1"].unique()
    }

    result_arr = np.zeros(len(traj), dtype=int)
    index_dict = defaultdict(list)
    for i, node in enumerate(traj["node_id1"].values):
        index_dict[node].append(i)

    for node, idx_list in index_dict.items():
        dep_times = dep_times_per_node.get(node)
        if dep_times is None or dep_times.size == 0:
            continue
        idx_arr = np.fromiter(idx_list, dtype=int)
        platform_arrival_ts = traj[platform_arrival_ts_col].values[idx_arr] - 1
        board_ts = traj["board_ts"].values[idx_arr]
        
        # TODO: check logic: dep_times=[0,10,20,30] -> {(arr10, board20): 1, (arr5, board20): 1, (arr10, board30): 2}
        idx_start = np.searchsorted(dep_times, platform_arrival_ts, side='right')
        idx_board = np.searchsorted(dep_times, board_ts, side='left')
        result_arr[idx_arr] = idx_board - idx_start

    traj["left_behind_count"] = result_arr
    return traj


def add_platform_arrival_ts_to_traj(traj: pd.DataFrame, use_egress_time_percentage: bool = False) -> pd.DataFrame:
    """
    Add walk times and corresponding platform_arrival_ts columns to trajectory dataframe.

    Added cols: [pp_id1, pp_id2, walk_time_percentage, walk_time, platform_arrival_ts]

    Optionally use egress walk time percentage as entry walk time percentage.
    If not, use random walk time percentage.

    Should the walk time + prev_ts = platform_arrival_ts > board_ts, set platform_arrival_ts = board_ts.

    :param traj: dataframe of trajectory with `prev_ts` column added. 
        Should be generated from `add_cols_to_traj` function.

        Expected columns (at least): [rid, node_id1, node_id2, board_ts, alight_ts, prev_ts]
    :type traj: pd.DataFrame
    :param use_egress_time_percentage: if True, use egress walk time percentage as entry walk time percentage. If False, use random walk time percentage.
    :type use_egress_time_percentage: bool, optional, default=False
    :return: traj with added columns
    :rtype: pd.DataFrame
    """
    # add [pp_id1, pp_id2] to traj
    plats1 = pd.DataFrame(get_platform()[:, [0, 1]], columns=[
                          "pp_id1", "node_id"])
    plats2 = pd.DataFrame(get_platform()[:, [0, 1]], columns=[
                          "pp_id2", "node_id"])
    traj = pd.merge(
        traj, plats1, left_on="node_id1", right_on="node_id", how="left"
    ).drop(columns=["node_id"]).merge(
        plats2, left_on="node_id2", right_on="node_id", how="left"
    ).drop(columns=["node_id"])

    # add entry walk time / transfer walk time to traj
    etd = get_etd()  # pp_id, x, pdf, cdf
    pp_id_to_cdf = {
        int(pp_id):  # key: pp_id
        etd[etd[:, 0] == pp_id][:, -1]  # value: array of [cdf]
        for pp_id in np.unique(etd[:, 0])
    }

    # TODO: assume all walk links are entry links (no transfer links... just for ease of use)
    if use_egress_time_percentage:
        afc = pd.DataFrame(get_afc()[:, [0, -1]],
                           columns=["rid", "ts2"]).set_index("rid")
        df_egress = pd.DataFrame({
            "egress_time": (afc['ts2'] - traj.groupby("rid")["alight_ts"].last()).dropna(),
            "pp_id2": traj.groupby("rid")["pp_id2"].last()
        })
        df_egress['walk_time_percentage'] = [
            pp_id_to_cdf[pp_id][int(time)] if (
                pp_id in pp_id_to_cdf and time <= 500) else 1
            for pp_id, time in zip(df_egress['pp_id2'], df_egress['egress_time'])
        ]
        traj['walk_time_percentage'] = traj['rid'].map(
            df_egress["walk_time_percentage"])
    else:
        traj['walk_time_percentage'] = np.random.rand(
            len(traj))  # randomly 0-1 value

    # 2️⃣ 创建结果 array
    walk_time_arr = np.zeros(len(traj), dtype=int)

    # 3️⃣ 按 pp_id 批处理
    for pp_id, cdf in pp_id_to_cdf.items():
        # 找出这个 pp_id 的行索引
        mask = traj['pp_id1'] == pp_id
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            continue  # skip 没有这个 pp_id 的

        percentages = traj.loc[indices, 'walk_time_percentage'].values
        walk_time = np.searchsorted(cdf, percentages)

        walk_time_arr[indices] = walk_time

    # 4️⃣ 写回 DataFrame
    traj['walk_time'] = walk_time_arr
    traj['platform_arrival_ts'] = traj['prev_ts'] + traj['walk_time']
    traj.loc[traj['platform_arrival_ts'] >=
             traj['board_ts'], 'platform_arrival_ts'] = traj['board_ts']
    
    return traj


def count_in_station_psg(traj: pd.DataFrame, node_id: int, t_width:int=10) -> tuple[np.ndarray, np.ndarray]:
    """
    Count in-station passengers at given node_id using traj dataframe.
    traj must have columns: ['node_id1', 'prev_ts', 'board_ts']
    """
    traj_filtered = traj[traj['node_id1'] == node_id].copy()
    enter_time = traj_filtered['prev_ts'].to_numpy()
    leave_time = traj_filtered['board_ts'].to_numpy()

    mask_valid = (~np.isnan(enter_time)) & (~np.isnan(leave_time))
    enter_time = enter_time[mask_valid]
    leave_time = leave_time[mask_valid]

    enter_time.sort()
    leave_time.sort()

    times = np.arange(21000, 86000, t_width)
    n_entered = np.searchsorted(enter_time, times, side='right')
    n_left = np.searchsorted(leave_time, times, side='right')
    in_station_psgs = n_entered - n_left

    return times, in_station_psgs


def count_platform_waiting_psg_with_distribution(traj:pd.DataFrame, node_id=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Count time-series of platform waiting passengers using distributed entry time CDFs.
    
    Each passenger is conceptually divided into 500 fractional units, which probabilistically 
    arrive at the platform according to a predefined cumulative distribution function (CDF). 
    This simulates the temporal dispersion of walking behavior from gate to platform.

    This function assumes the input `traj` contains basic structural columns like
    ['path_id', 'seg_id', 'prev_ts', 'board_ts', 'node_id1', ...],
    which should be added via `add_cols_to_traj()` beforehand.
    
    :param traj: trajectory DataFrame with necessary structural fields.
    :type traj: pd.DataFrame
    :param node_id: target node_id (platform). If None, will return 0s.
    :type node_id: int or None
    :return: 
        - times: time points (from 21000 to 86000 seconds, step=1)
        - waiting_psg: number of passengers waiting on the platform at each second
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    wtdm = WalkTimeDisModel()
    
    # determine node_id
    platform = get_platform()  # pp_id, node_id, uid
    if node_id is None:
        node_id = np.random.choice(platform[:, 1], 1)[
            0] if node_id is None else node_id
        pp_id, node_id, uid = platform[platform[:, 1] == node_id][0]
        is_transfer = (read_("STA", show_timer=False, drop_cols=True,
                    quiet_mode=True)["STATION_UID"] == uid).sum() > 1
        print(
            f"node_id: {node_id} UID: {uid} pp_id: {pp_id}, is transfer: {is_transfer}")
    else:
        pp_id, node_id, uid = platform[platform[:, 1] == node_id][0]

    traj = traj[traj["node_id1"] == node_id]

    map_df = wtdm.path_seg2pp_id_mima.copy()
    map_df['seg_id_map'] = map_df['seg_id'] + 1
    traj = traj.merge(
        map_df[['path_id', 'seg_id_map', 'pp_id_min', 'pp_id_max']],
        left_on=['path_id', 'seg_id'],
        right_on=['path_id', 'seg_id_map'],
        how='left'
    )

    # Assign the appropriate CDF array for each row (entry or transfer)
    def get_cdf(row):
        if row['seg_id'] == 1:
            return wtdm.pp_id2cdf_table.get(pp_id, None)
        key = (row['pp_id_min'], row['pp_id_max'])
        return wtdm.transfer_mima2cdf_table.get(key, None)
    traj["cdf_array"] = traj.apply(get_cdf, axis=1)

    # Prepare time grid and output array
    times = np.arange(21000, 86000, 1)  # Every second from 5:50 to ~24:00
    in_station_psgs = np.zeros_like(times, dtype=float)

    # Extract key variables for vectorized iteration
    start_idx = np.searchsorted(times, traj['prev_ts'].values)
    duration = (traj['board_ts'] -
                traj['prev_ts']).astype(int).values
    cdfs = traj['cdf_array'].values

    indices_list = []
    values_list = []

    for i in range(len(traj)):
        dur = duration[i]
        cdf = cdfs[i]
        idx_start = start_idx[i]

        if dur <= 0 or idx_start >= len(times):
            print(f"Invalid duration or start index: {dur}, {idx_start}")
            print(f"traj.iloc[{i}]: {traj.iloc[i]}")
            continue

        max_cdf_idx = len(cdf) - 1
        effective_len = min(dur, max_cdf_idx)
        idx_end = min(idx_start + effective_len, len(times) - 1)

        idx_range = np.arange(idx_start, idx_end + 1)
        if dur >= len(cdf):
            val = np.ones_like(idx_range)
        else:
            val = cdf[:len(idx_range)]

        indices_list.append(idx_range)
        values_list.append(val)

    all_indices = np.concatenate(indices_list)
    all_values = np.concatenate(values_list)
    np.add.at(in_station_psgs, all_indices, all_values)

    return times, in_station_psgs
