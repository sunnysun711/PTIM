"""
itinerary_probability.py

A module for calculating walking time distributions, train segment penalties, and
final itinerary selection probabilities in a metro passenger itinerary analysis.

This module provides a processing pipeline for feasible itineraries by:
- Computing entry, egress, and transfer walking time distributions
- Saving and filtering distribution files for targeted itineraries (left)
- Assigning segment-level penalties based on train load
- Calculating itinerary-level probabilities combining walk distributions and penalties

Key functionalities:
--------------------
Main function (recommended for users):
- compute_itinerary_probabilities(): Calculate final itinerary selection probabilities

Supporting public functions:
- attach_walk_dis_all(): Compute and attach entry, egress, transfer walk distributions to all itineraries
- filter_dis_file(): Filter a saved distribution file to include only itineraries in `left`
- cal_in_vehicle_penal_all(): Map penalties onto in-vehicle segments

Internal helper functions:
- _attach_entry_dis_all(): Attach entry walk time CDF to entry links of all itineraries.
- _attach_egress_dis_all(): Attach egress walk time PDF to egress links of all itineraries.
- _attach_transfer_dis_all(): Attach transfer walk time CDF to transfer links of all itineraries.

Typical workflow:
-----------------
1. First-time preparation:
    - Run `attach_walk_dis_all()` to compute all walk time distributions
    - Save the result to file (e.g., `save_("dis.pkl", dis_df)`)
2. For each assignment session:
    - Load and filter the saved distribution: `dis_df_from_file = filter_dis_file()`
    - Run `cal_in_vehicle_penal_all()` to assign penalties
    - Run `compute_itinerary_probabilities()` to calculate itinerary probabilities

Advantages:
-----------
- Avoids recomputing walk distributions for every assignment
- Keeps a consistent pipeline: always load -> filter -> calculate probabilities

Dependencies:
-------------
- Requires precomputed walk time distribution models (WalkTimeDisModel)
- Input data from AFC, platform mappings, and left itineraries

Example usage:
--------------
>>> # Step 1: Calculate and save walk distributions (one-time preparation)
>>> dis_df = attach_walk_dis_all()
>>> save_(config.CONFIG["results"]["dis"], dis_df)

>>> # Step 2: Filter walk distribution file to current left itineraries
>>> dis_df_from_file = filter_dis_file()

>>> # Step 3: Build penalty mapping DataFrame from overload sections
>>> overload_train_section = find_overload_train_section()  # from src.timetable
>>> penal_mapper_df = build_penal_mapper_df(overload_train_section)  # from src.congest_penal

>>> # Step 4: Map penalties onto itineraries
>>> penal_df = cal_in_vehicle_penal_all(penal_mapper_df)

>>> # Step 5: Calculate final itinerary probabilities
>>> prob_df = compute_itinerary_probabilities(dis_df_from_file, penal_df)
"""


import numpy as np
import pandas as pd

from src import config
from src.utils import read_
from src.globals import get_k_pv, get_platform, get_afc, get_etd, get_ttd
from src.walk_time_filter import get_path_seg_to_pp_ids, get_transfer_from_feas_iti
from src.walk_time_dis_calculator import WalkTimeDisModel


def _attach_entry_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the entry walk distribution and attach them to all left itineraries.

    :param wtd: (optional) WalkTimeDisModel instance.
        Defaults to None (automatically created with the latest etd, ttd CSV files).
    :type wtd: WalkTimeDisModel, optional

    :param left: (optional) DataFrame of left itineraries.
        Defaults to None (read from left.pkl file).

        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`]
    :type left: pd.DataFrame, optional
        
    
    :return: DataFrame of left itineraries with entry walk distribution columns attached:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `board_ts`: The boarding timestamp.
        - `entry_pp_id`: The physical platform ID of the entry platform.
        - `ts1`: The tap-in time of the rid.
        - `entry_time`: The entry walking time. (plus waiting time)
        - `dis`: The distribution value of the entry time.
        
    :rtype: pd.DataFrame
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for _attach_entry_dis()...")
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


def _attach_egress_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the egress walk distribution and attach them to all left itineraries.

    :param wtd: (optional) WalkTimeDisModel instance.
        Defaults to None (automatically created with the latest etd, ttd CSV files).
    :type wtd: WalkTimeDisModel, optional

    :param left: (optional) DataFrame of left itineraries.
        Defaults to None (read from left.pkl file).
        
        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`]
    :type left: pd.DataFrame, optional

    :return: DataFrame of left itineraries with egress walk distribution columns attached:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `alight_ts`: The alighting timestamp.
        - `egress_pp_id`: The physical platform ID of the egress platform.
        - `ts2`: The tap-out time of the rid.
        - `egress_time`: The egress walking time.
        - `dis`: The distribution value of the egress time.
        
    :rtype: pd.DataFrame
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for _attach_egress_dis()...")
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

    return left_last_seg


def _attach_transfer_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the transfer walk distribution and attach them to all itineraries in `left`.

    :param wtd: (optional) WalkTimeDisModel instance.
        Defaults to None (automatically created with the latest etd, ttd CSV files).
    :type wtd: WalkTimeDisModel, optional

    :param left: (optional) DataFrame of left itineraries.
        Defaults to None (read from `left.pkl` file).
        
        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`].
    :type left: pd.DataFrame, optional

    :returns: DataFrame of itineraries with transfer walk distribution columns attached:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `seg_id`: The segment ID. (seg_id of the alighting train)
        - `train_id`: The ID of the train.
        - `board_ts`: The boarding timestamp.
        - `alight_ts`: The alighting timestamp.
        - `transfer_time`: The calculated transfer time for the transfer link.
        - `transfer_type`: The type of transfer (e.g., platform to platform).
        - `pp_id_min`: The minimum physical platform ID.
        - `pp_id_max`: The maximum physical platform ID.
        - `dis`: The distribution value of the transfer time.

    :rtype: pd.DataFrame
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for _attach_transfer_dis()...")
        wtd = WalkTimeDisModel(etd=get_etd(), ttd=get_ttd())
    if left is None:
        left = read_(config.CONFIG["results"]["left"], show_timer=False)

    # Extract path segments with transfer
    # ['rid', 'iti_id', 'path_id', 'seg_id', 'alight_ts', 'board_ts', 'transfer_time']
    df = get_transfer_from_feas_iti(df_feas_iti=left)

    # ["path_id", "seg_id", "pp_id1", "pp_id2", "transfer_type"]
    df_ps2pp = get_path_seg_to_pp_ids()
    df_ps2pp["pp_id_min"] = df_ps2pp[["pp_id1", "pp_id2"]].min(axis=1)
    df_ps2pp["pp_id_max"] = df_ps2pp[["pp_id1", "pp_id2"]].max(axis=1)
    df_ps2pp.drop(columns=["pp_id1", "pp_id2"], inplace=True)

    # Merge path segments with transfer and path segments to physical platforms
    df = df.merge(df_ps2pp, on=["path_id", "seg_id"], how="left")
    df['path_id'] = df['path_id'].astype(int)

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

    return df


def attach_walk_dis_all(wtd: WalkTimeDisModel = None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the entry, egress, and transfer walk distributions and attach them to all itineraries in `left`.

    Approximately 33 seconds for the left with 46,637,884 itineraries. (PC)
    
    This is a one-timer function. No need to calculate distribution values every time of assignment.
    
    So saving the distribution values to a file is recommended.
    
    Recommended usage:
    ```python
    dis_df = attach_walk_dis_all()
    save_(config.CONFIG["results"]["dis"], dis_df)
    ```

    :param wtd: (optional) WalkTimeDisModel instance.
        Defaults to None (automatically created with the latest etd, ttd CSV files).
    :type wtd: WalkTimeDisModel, optional
    
    :param left: (optional) DataFrame of left itineraries.
        Defaults to None (read from left.pkl file).
        
        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`].
    :type left: pd.DataFrame, optional

    :returns: DataFrame of entry, egress, and transfer walk distributions for all itineraries.
        
        Each row is a walk link with the following columns:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `seg_id`: The segment ID. (in_vehicle seg_id for transfer, 0 for entry, -1 for egress)
        - `t1`: The starting timestamp of the walk link.
        - `t2`: The ending timestamp of the walk link.
        - `pp_id1`: pp_id_min for transfer, to-board pp_id for entry, 0 for egress.
        - `pp_id2`: pp_id_max for transfer, 0 for entry, alighted pp_id for egress.
        - `time`: Full walk link time. (including platform waiting time)
        - `dis`: The distribution value of the time.
        
    :rtype: pd.DataFrame
    """
    if wtd is None:
        print("[INFO] Initializing WalkTimeDisModel for attach_walk_dis_all()...")
        wtd = WalkTimeDisModel(etd=get_etd(), ttd=get_ttd())
    if left is None:
        left = read_(config.CONFIG["results"]["left"], show_timer=False)

    print("[INFO] Attaching entry walk distribution for all itineraries...")
    df_ent = _attach_entry_dis_all(wtd=wtd, left=left)
    df_ent["seg_id"] = 0  # set entry seg_id to 0
    df_ent["pp_id2"] = 0  # set entry pp_id2 to 0
    df_ent = df_ent.rename(
        columns={
            "ts1": "t1",
            "board_ts": "t2",
            "entry_time": "time",
            "entry_pp_id": "pp_id1"}
    )[[
        "rid", 'iti_id', 'path_id', 'seg_id', 't1', 't2', 'pp_id1', 'pp_id2', 'time', 'dis'
    ]]

    print("[INFO] Attaching egress walk distribution for all itineraries...")
    df_egr = _attach_egress_dis_all(wtd=wtd, left=left)
    df_egr["seg_id"] = -1  # set egress seg_id to -1
    df_egr["pp_id1"] = 0  # set egress pp_id1 to 0
    df_egr = df_egr.rename(
        columns={
            "alight_ts": "t1",
            "ts2": "t2",
            "egress_time": "time",
            "egress_pp_id": "pp_id2",
        }
    )[[
        "rid", "iti_id", "path_id", "seg_id", "t1", "t2", "pp_id1", "pp_id2", "time", "dis"
    ]]

    print("[INFO] Attaching transfer walk distribution for all itineraries...")
    df_trans = _attach_transfer_dis_all(wtd=wtd, left=left)
    df_trans = df_trans.rename(
        columns={
            "pp_id_min": "pp_id1",
            "pp_id_max": "pp_id2",
            "transfer_time": "time",
            "alight_ts": "t1",
            "board_ts": "t2"
        }
    )[[
        "rid", "iti_id", "path_id", "seg_id", "t1", "t2", "pp_id1", "pp_id2", "time", "dis"
    ]]

    df = pd.concat([df_trans, df_ent, df_egr], ignore_index=True)
    return df


def filter_dis_file(dis_df_from_file:pd.DataFrame=None, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Filter the distribution file to only include the itineraries in `left`.

    :param dis_df_from_file: DataFrame of distribution values attached itineraries read from file.
        Defaults to None (read from `dis.pkl` file).
    :type dis_df_from_file: pd.DataFrame, optional, default=None
    
    :param left: DataFrame of left itineraries.
        Defaults to None (read from `left.pkl` file).

        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`].
    :type left: pd.DataFrame, optional, default=None
    
    :return: Filtered DataFrame of distribution values for itineraries in `left`.
    
        Each row is a walk link with the following columns:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `seg_id`: The segment ID. (in_vehicle seg_id for transfer, 0 for entry, -1 for egress)
        - `t1`: The starting timestamp of the walk link.
        - `t2`: The ending timestamp of the walk link.
        - `pp_id1`: pp_id_min for transfer, to-board pp_id for entry, 0 for egress.
        - `pp_id2`: pp_id_max for transfer, 0 for entry, alighted pp_id for egress.
        - `time`: Full walk link time. (including platform waiting time)
        - `dis`: The distribution value of the time.
    :rtype: pd.DataFrame
    """
    if dis_df_from_file is None:
        dis_df_from_file = read_(config.CONFIG["results"]["dis"], show_timer=False)
    if left is None:
        left = read_(config.CONFIG["results"]["left"], show_timer=False)
        
    dis_df_from_file = dis_df_from_file[dis_df_from_file["rid"].isin(left["rid"].unique())]
    return dis_df_from_file


def cal_in_vehicle_penal_all(penal_mapper_df: pd.DataFrame, left: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate in_vehicle penalties for all itineraries in `left` based on `penal_mapper_df`.

    :param penal_mapper_df: A DataFrame mapping a tuple of (train_id, board_ts, alight_ts) 
        to a penalty value (float between 0 and 1).
        
        Expected columns: [`train_id`, `board_ts`, `alight_ts`, `penalty`]
        
        This variable should be obtained from `src.congest_penal.build_penalty_df()`.
    
    :type penal_mapper_df: pd.DataFrame

    :param left: (optional) DataFrame of left itineraries.
        Defaults to None (read from `left.pkl` file).
        
        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`].
        
    :type left: pd.DataFrame, optional, default=None

    :returns: DataFrame of left itineraries with an added penalty column including the following columns:
        
        - `rid`: The ID of the transaction record.
        - `iti_id`: The itinerary ID.
        - `path_id`: The path ID.
        - `seg_id`: The segment ID. (correspond to in_vehicle links)
        - `train_id`: The ID of the train.
        - `board_ts`: The boarding timestamp.
        - `alight_ts`: The alighting timestamp.
        - `penalty`: The calculated penalty value for the segment.
        
    :rtype: pd.DataFrame
    """
    if left is None:
        left = read_(config.CONFIG["results"]["left"], show_timer=False)

    penalized_iti_df = pd.merge(left=left, right=penal_mapper_df, on=["train_id", "board_ts", "alight_ts"], how="left")
    penalized_iti_df["penalty"].fillna(1.0, inplace=True)

    return penalized_iti_df


def compute_itinerary_probabilities(dis_attached_iti: pd.DataFrame, penalized_iti: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probabilities for all feasible itineraries with the provided
    walk time distributions and train segment penalties.

    :param dis_attached_iti: DataFrame of itineraries with walk time distributions attached. 
        
        Columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `t1`, `t2`, `pp_id1`, `pp_id2`, `time`, `dis`]
        
        Should be generated by either:
            - `src.itinerary.attach_walk_dis_all()`
            - `src.itinerary.filter_dis_file()`
    :type dis_attached_iti: pd.DataFrame
    
    :param penalized_iti: DataFrame of penalized itineraries with columns:

        [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`, `penalty`]

        Should be generated by `src.itinerary.cal_in_vehicle_penal_all()`.
    :type penalized_iti: pd.DataFrame
    
    :returns: DataFrame of probabilities for all feasible itineraries with columns:

        [`rid` (index), `iti_id` (index), `dis`, `penalty`, `dis_penaled`, `dis_penaled_sum`, `prob`]

        The index is a MultiIndex of (`rid`, `iti_id`).
    :rtype: pd.DataFrame
    """
    # index: (rid, iti_id) columns: dis
    dis_prod = dis_attached_iti.groupby(["rid", "iti_id"])["dis"].prod()
    # index: (rid, iti_id) columns: penalty
    penalty_prod = penalized_iti.groupby(["rid", "iti_id"])["penalty"].prod()
    # Combine the two Series into a DataFrame
    df = pd.concat([dis_prod, penalty_prod], axis=1)
    df["dis_penaled"] = df["dis"] * df["penalty"]
    df["dis_penaled_sum"] = df.groupby(level=[0])["dis_penaled"].transform("sum")
    df["prob"] = df["dis_penaled"] / df["dis_penaled_sum"]
    return df  # index (rid, iti_id) columns: [dis, penalty, dis_penaled, dis_penaled_sum, prob]


if __name__ == "__main__":
    import time
    config.load_config()
    a = time.time()
    dis = attach_walk_dis_all()
    dis.groupby(["rid", "iti_id"])["dis"].prod()
    print(time.time() - a)
