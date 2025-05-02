# calculate itinerary probabilities, save in iti_prob_*.pkl
import numpy as np
import pandas as pd

from src import config
from src.utils import read_, read_all
from src.globals import get_k_pv, get_platform, get_afc
from src.walk_time_filter import get_path_seg_to_pp_ids, get_transfer_from_feas_iti
from src.walk_time_dis_calculator import WalkTimeDisModel


def cal_entry_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the entry walk distribution for all itineraries in `left`.

    Args:
        wtd (WalkTimeDisModel, optional): WalkTimeDisModel instance.
            Defaults to None (automatically created with the latest etd, ttd CSV files).
        left (pd.DataFrame, optional): DataFrame of left itineraries.
            Defaults to None (read from left.pkl file).
            Expected columns: ['rid', 'iti_id', 'path_id', 'seg_id', 'train_id', 'board_ts', 'alight_ts']

    Returns:
        pd.DataFrame: DataFrame of entry walk distribution for all itineraries.
            columns are: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts, entry_pp_id, ts1, entry_time, dis]
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for cal_entry_dis()...")
        wtd = WalkTimeDisModel(etd=get_etd(), ttd=get_ttd())
    if left is None:
        left = read_(config.CONFIG["results"]["left"])

    # Extract first segment per itinerary
    left_first_seg = left[left["seg_id"] == 1][[
        "rid", "iti_id", "path_id", "board_ts"]].copy()

    # Map path_id → entry platform physical platform id (pp_id)
    k_pv = get_k_pv()[:, :4]  # ["path_id", "pv_id", "node_id1", "node_id2"]
    entry_ids = np.where(k_pv[:, 1] == 1)[0]
    path_entry = k_pv[entry_ids][:, [0, -1]]  # [path_id, node_id (platform)]

    node_id2pp_id = {i: j for j, i, _ in get_platform()}
    path_entry_pp_id = np.hstack([path_entry, np.vectorize(
        node_id2pp_id.get)(path_entry[:, 1]).reshape(-1, 1)])
    path_id2entry_pp_id = {path_id: entry_pp_id for path_id,
                           entry_pp_id in path_entry_pp_id[:, [0, 2]]}

    # Map entry physical platform id
    left_first_seg["entry_pp_id"] = left_first_seg["path_id"].map(
        path_id2entry_pp_id)

    # Calculate entry times relative to AFC tap-in times
    afc_ts1 = pd.DataFrame(get_afc()[:, [0, 2]], columns=[
        "rid", "ts1"]).set_index("rid")
    left_first_seg["ts1"] = left_first_seg["rid"].map(afc_ts1["ts1"])
    left_first_seg["entry_time"] = left_first_seg["board_ts"] - \
        left_first_seg["ts1"]

    # Calculate entry walk distribution for each physical platform group
    left_first_seg["dis"] = 0.0  # to be filled
    for entry_pp_id, df_ps2pp_ in left_first_seg.groupby("entry_pp_id"):
        # print(entry_pp_id, df_ps2pp_.shape)
        df_ps2pp_["dis"] = wtd.compute_entry_cdf_from_pp(pp_id=entry_pp_id, times_start=np.zeros(
            df_ps2pp_.shape[0]), times_end=df_ps2pp_["entry_time"].values)
        left_first_seg.loc[df_ps2pp_.index, "dis"] = df_ps2pp_["dis"].values

    # columns of left_first_seg: [rid, iti_id, path_id, board_ts, entry_pp_id, ts1, entry_time, dis]
    return left_first_seg


def cal_egress_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the egress walk distribution for all left itineraries.

    Args:
        wtd (WalkTimeDisModel, optional): WalkTimeDisModel instance. Defaults to None.
            If wtd is None, create a new instance with the lastest etd, ttd csv files.
        left (pd.DataFrame, optional): left dataframe. Defaults to None.
            columns are: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts]
            If left is None, read from the left.pkl file.

    Returns:
        pd.DataFrame: egress walk distribution for all left itineraries.
            columns are: [rid, iti_id, path_id, alight_ts, egress_pp_id, ts2, egress_time, dis]
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for cal_egress_dis()...")
        wtd = WalkTimeDisModel(etd=get_etd(), ttd=get_ttd())
    if left is None:
        left = read_(config.CONFIG["results"]["left"])

    # Extract last segment per itinerary
    left_last_seg = left[np.append(left["seg_id"].values[1:] == 1, True)][[
        "rid", "iti_id", "path_id", "alight_ts"]].copy()

    # Map path_id → egress platform physical platform id (pp_id)
    k_pv = get_k_pv()[:, :4]  # ["path_id", "pv_id", "node_id1", "node_id2"]
    egress_ids = np.where(k_pv[:, 1] == 1)[0] - 1
    path_egress = k_pv[egress_ids][:, [0, -2]]  # [path_id, node_id (platform)]

    node_id2pp_id = {i: j for j, i, _ in get_platform()}
    path_egress_pp_id = np.hstack([path_egress, np.vectorize(
        node_id2pp_id.get)(path_egress[:, 1]).reshape(-1, 1)])
    path_id2egress_pp_id = {path_id: egress_pp_id for path_id,
                            egress_pp_id in path_egress_pp_id[:, [0, 2]]}

    # Map egress physical platform id
    left_last_seg["egress_pp_id"] = left_last_seg["path_id"].map(
        path_id2egress_pp_id)

    # Calculate egress times relative to AFC tap-out times
    afc_ts2 = pd.DataFrame(get_afc()[:, [0, 4]], columns=[
        "rid", "ts2"]).set_index("rid")
    left_last_seg["ts2"] = left_last_seg["rid"].map(afc_ts2["ts2"])
    left_last_seg["egress_time"] = left_last_seg["ts2"] - \
        left_last_seg["alight_ts"]

    # Calculate egress walk distribution for each physical platform group
    left_last_seg["dis"] = 0.0  # to be filled
    for egress_pp_id, df_ps2pp_ in left_last_seg.groupby("egress_pp_id"):
        # print(egress_pp_id, df_ps2pp_.shape)
        df_ps2pp_["dis"] = wtd.compute_egress_pdf_from_pp(
            pp_id=egress_pp_id, times=df_ps2pp_["egress_time"].values)
        left_last_seg.loc[df_ps2pp_.index, "dis"] = df_ps2pp_["dis"].values

    # columns of left_last_seg: [rid, iti_id, path_id, alight_ts, egress_pp_id, ts2, egress_time, dis]
    return left_last_seg


def cal_transfer_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the transfer walk distribution for all itineraries in `left`.

    Args:
        wtd (WalkTimeDisModel, optional): WalkTimeDisModel instance.
            Defaults to None (automatically created with the latest etd, ttd CSV files).
        left (pd.DataFrame, optional): DataFrame of left itineraries.
            Defaults to None (read from left.pkl file).
            Expected columns: ['rid', 'iti_id', 'path_id','seg_id', 'train_id', 'board_ts', 'alight_ts']

    Returns:
        pd.DataFrame: DataFrame of transfer walk distribution for all itineraries.
            columns are: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts, transfer_time, transfer_type, pp_id_min, pp_id_max, dis]
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for cal_transfer_dis()...")
        wtd = WalkTimeDisModel(etd=get_etd(), ttd=get_ttd())
    if left is None:
        left = read_(config.CONFIG["results"]["left"], show_timer=False)

    # Extract path segments with transfer
    # ['rid', 'iti_id', 'path_id', 'seg_id', 'alight_ts', 'transfer_time']
    df = get_transfer_from_feas_iti(df_feas_iti=left)

    # ["path_id", "seg_id", "pp_id1", "pp_id2", "transfer_type"]
    df_ps2pp = get_path_seg_to_pp_ids()
    df_ps2pp["pp_id_min"] = df_ps2pp[["pp_id1", "pp_id2"]].min(axis=1)
    df_ps2pp["pp_id_max"] = df_ps2pp[["pp_id1", "pp_id2"]].max(axis=1)
    df_ps2pp.drop(columns=["pp_id1", "pp_id2"], inplace=True)

    # Merge path segments with transfer and path segments to physical platforms
    df = df.merge(df_ps2pp, on=["path_id", "seg_id"], how="left")

    # calculate transfer time CDF for each physical platform group
    df["dis"] = 0.0  # to be filled
    for (pp_id_min, pp_id_max), df_ in df.groupby(["pp_id_min", "pp_id_max"]):
        # print(pp_id_min, pp_id_max, df_.shape)
        dis = wtd.compute_transfer_cdf_from_pp(
            pp_id_min, pp_id_max,
            times_start=np.zeros(df_.shape[0]),
            times_end=df_["transfer_time"].values
        )
        df.loc[df_.index, "dis"] = dis
    # columns of df: [rid, iti_id, path_id, seg_id, alight_ts, transfer_time, transfer_type, pp_id_min, pp_id_max, dis]
    return df
