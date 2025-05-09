from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.globals import get_afc, get_k_pv, get_platform, get_tt
from src.utils import read_all


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


def add_cols_to_traj(config_file: str) -> pd.DataFrame:
    """
    add left-behind-times columns to trajectory dataframe.
    
    Specifically: [node_id1, node_id2, line, updown, prev_ts, left_behind_count]

    :param config_file: config file name, example: `config1`
    :type config_file: str
    :return: DataFrame with ['rid', 'iti_id', 'path_id', 'seg_id', 'train_id', 'board_ts',
       'alight_ts', 'node_id1', 'node_id2', 'line', 'updown', 'prev_ts', 'left_behind_count']
    :rtype: pd.DataFrame
    """
    config.load_config(f"configs/{config_file}.yaml")

    traj = read_all(config.CONFIG["results"]["assigned"], show_timer=False)
    afc = get_afc()
    k_pv = add_seg_id_to_k_pv(get_k_pv())
    tt = get_tt()

    df_k_pv = pd.DataFrame(k_pv, columns=["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown", "seg_id"])
    traj = traj.merge(df_k_pv[["path_id", "seg_id", "node_id1", "node_id2", "line", "updown"]], on=["path_id", "seg_id"], how='left')
    traj["prev_ts"] = traj["alight_ts"].shift(1)

    df_rid_ts1 = pd.DataFrame(afc[:, [0, 2]], columns=["rid", "ts1"])
    traj = traj.merge(df_rid_ts1, on="rid", how="left")
    traj.loc[traj["seg_id"] == 1, "prev_ts"] = traj.loc[traj["seg_id"] == 1, "ts1"]

    dep_times_per_node = {
        node: np.sort(tt[(tt[:, 1] == node // 10) & (tt[:, 3] == (1 if node % 10 == 0 else -1))][:, 4])
        for node in traj["node_id1"].unique()
    }
    print(f"Prepared dep_times for {len(dep_times_per_node)} nodes")

    result_arr = np.zeros(len(traj), dtype=int)
    index_dict = defaultdict(list)
    for i, node in enumerate(traj["node_id1"].values):
        index_dict[node].append(i)

    for node, idx_list in index_dict.items():
        dep_times = dep_times_per_node.get(node)
        if dep_times is None or dep_times.size == 0:
            continue
        idx_arr = np.fromiter(idx_list, dtype=int)
        prev_ts = traj["prev_ts"].values[idx_arr]
        board_ts = traj["board_ts"].values[idx_arr]
        result_arr[idx_arr] = np.searchsorted(dep_times, board_ts, side='left') - np.searchsorted(dep_times, prev_ts, side='left')

    traj["left_behind_count"] = result_arr
    traj["prev_ts"] = traj["prev_ts"].astype(int)
    traj.drop(columns=["ts1"], inplace=True)
    return traj
