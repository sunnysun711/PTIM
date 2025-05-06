"""
This module is dedicated to filtering and processing egress and transfer time data
from various data sources, specifically `left.pkl` and `assigned_*.pkl` files. It
plays a crucial role in preparing the data for further analysis and visualization
related to passenger trajectories.

Key functionalities include:
- Extracting egress times from different data sources and mapping them to physical 
    platform IDs.
- Calculating egress times based on the difference between alighting and tap-out 
    times.
- Validating data consistency to ensure accurate results.
- Extracting transfer times and mapping them to physical platform ID pairs.
- Processing and aggregating data to obtain essential information for analysis.

The module provides several utility functions that handle different aspects of data
extraction, calculation, and filtering. These functions are designed to be modular
and reusable, facilitating the overall data processing pipeline.

Functions:
    get_egress_from_left(): Extract egress times from left.pkl.
    get_egress_from_assigned(): Extract egress times from assigned_*.pkl.
    _calculate_egress_time(): Calculate egress time for the given last-seg DataFrame.
    filter_egress_all(): Combine and filter all egress time data.
    get_transfer_from_assigned(): Extract transfer times from assigned_*.pkl.
    get_path_seg_to_pp_ids(): Map path_id and seg_id to physical platform IDs.
    filter_transfer_all(): Combine and filter all transfer time data.
"""
import numpy as np
import pandas as pd

from src import config
from src.utils import read_, read_all
from src.globals import get_afc, get_k_pv, get_platform


def get_egress_from_left() -> pd.DataFrame | None:
    """
    Find rids in left.pkl where all feasible itineraries share the same final train_id.
    In this case, all itineraries of that rid lead to the same egress time.

    :return: 
        DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
        If left is empty, return None.
        It includes the following columns:

        - `node1`: The starting node of the egress path.
        - `node2`: The ending node of the egress path.
        - `alight_ts`: The time when the passenger alighted from the vehicle.
        - `ts2`: The time when the passenger exited the station.
        - `egress_time`: The calculated egress time, which is the difference between `ts2` and `alight_ts`.
    :rtype: pd.DataFrame | None
    """
    df = read_(config.CONFIG["results"]["left"], show_timer=False)
    if df.empty:
        return None

    # Keep only the last segment for each itinerary of one rid
    last_seg_all_iti = df.groupby(["rid", "iti_id"]).last().reset_index()

    # filter rids with same train_ids in last seg (Line 7 issue is resolved by default)
    unique_end_count = last_seg_all_iti.groupby("rid")["train_id"].nunique()
    same_train_rids = unique_end_count[unique_end_count == 1].index

    # get last in_vehicle link  [rid (index), path_id, seg_id, train_id, alight_ts]
    # 120090 rows for first left.pkl file
    last_seg = last_seg_all_iti[
        last_seg_all_iti["rid"].isin(same_train_rids)
    ].groupby("rid").last().drop(columns=["iti_id", "board_ts"])

    return _calculate_egress_time(last_seg)


def get_egress_from_assigned() -> pd.DataFrame:
    """
    Find rids in all assigned_*.pkl files where all feasible itineraries share the same final train_id.

    :return: 
        DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
        It includes the following columns:
        
        - `node1`: The starting node of the egress path.
        - `node2`: The ending node of the egress path.
        - `alight_ts`: The time when the passenger alighted from the vehicle.
        - `ts2`: The time when the passenger exited the station.
        - `egress_time`: The calculated egress time, which is the difference between `ts2` and `alight_ts`.
    :rtype: pd.DataFrame
    """
    df = read_all(config.CONFIG["results"]["assigned"], show_timer=False)

    # get last in_vehicle link  [rid (index), path_id, seg_id, train_id, alight_ts]
    # 772934 rows for assigned_1.pkl file
    last_seg = df.groupby("rid").last().drop(columns=["iti_id", "board_ts"])

    return _calculate_egress_time(last_seg)


def _calculate_egress_time(df_last_seg: pd.DataFrame | None) -> pd.DataFrame:
    """
    A helper function designed to calculate the egress time for a specified set of passenger IDs (rids).
    It enriches the input DataFrame with necessary columns and computes the egress time for each passenger.
    The egress time is defined as the time difference between the passenger's exit time and alighting time.

    :param df_last_seg: A DataFrame that holds the last segment information of each itinerary.
                        Each row corresponds to a unique passenger, identified by the "rid" index.
                        If input None, returns an empty DataFrame.
    :type df_last_seg: pd.DataFrame | None

    :return: 
        DataFrame with the shape (n_rids, 5), where "rid" serves as the index.
        It includes the following columns:
        
        - `node1`: The starting node of the egress path.
        - `node2`: The ending node of the egress path.
        - `alight_ts`: The time when the passenger alighted from the vehicle.
        - `ts2`: The time when the passenger exited the station.
        - `egress_time`: The calculated egress time, which is the difference between `ts2` and `alight_ts`.
    :rtype: pd.DataFrame

    :raises AssertionError: If the last segment found does not match the last segment in the path,
                        indicating a potential data inconsistency.
    """
    if df_last_seg is None:
        return pd.DataFrame([], columns=["node1", "node2", "alight_ts", "ts2", "egress_time"])
    afc = get_afc()
    k_pv = get_k_pv()

    filtered_AFC = afc[np.isin(afc[:, 0], df_last_seg.index)]
    egress_link = k_pv[len(k_pv) - 1 - np.unique(k_pv[:, 0]
                                                 [::-1], return_index=True)[1], :4]
    path_id_node1 = {link[0]: link[2] for link in egress_link}
    path_id_node2 = {link[0]: link[3] for link in egress_link}
    df_last_seg['ts2'] = {record[0]: record[-1] for record in filtered_AFC}
    df_last_seg["UID2"] = {record[0]: record[-2] for record in filtered_AFC}
    df_last_seg["node1"] = df_last_seg["path_id"].map(path_id_node1)
    df_last_seg["node2"] = df_last_seg["path_id"].map(path_id_node2)
    assert df_last_seg[df_last_seg["node2"] != df_last_seg["UID2"]].shape[0] == 0, \
        "Last seg found is not the last seg in path!"
    df_last_seg['egress_time'] = df_last_seg['ts2'] - df_last_seg['alight_ts']
    return df_last_seg[["node1", "node2", "alight_ts", "ts2", "egress_time"]]


def filter_egress_all() -> pd.DataFrame:
    """
    Find egress times from left.pkl and assigned_*.pkl files, map them with physical_platform_id.

    :return: DataFrame with columns:
        ["rid"(index), "physical_platform_id", "alight_ts", "egress_time"]
        
        - physical_platform_id is the ID of the physical platform,
        - alight_ts is the time when the passenger alighted from the train,
        - egress_time is the time difference between ts2 and alight_ts.
    :rtype: pd.DataFrame
    """
    # [rid (index), node1, node2, alight_ts, ts2, egress_time]
    df = pd.concat([get_egress_from_left(), get_egress_from_assigned()])

    # [physical_platform_id, node_id, uid]
    platform = pd.DataFrame(get_platform(), columns=[
                            "physical_platform_id", "node_id", "uid"])

    df = df.reset_index().merge(platform, left_on="node1",
                                right_on="node_id", how="left").set_index("rid")
    assert df[df["uid"] != df["node2"]
              ].shape[0] == 0, "Egress UID not the same!"
    df = df[["physical_platform_id", "alight_ts", "egress_time"]]
    return df


def get_transfer_from_feas_iti(df_feas_iti: pd.DataFrame) -> pd.DataFrame:
    """
    Find transfer times from feasible itineraries.
    Input df_feas_iti could be left.pkl or assigned_*.pkl.

    :param df_feas_iti: DataFrame of feasible itineraries.
        Expected columns: ['rid', 'iti_id', 'path_id','seg_id', 'train_id', 'board_ts', 'alight_ts']
        
        **Important**: Please make sure the itineraries are sorted by ['rid', 'iti_id','seg_id']
    :type df_feas_iti: pd.DataFrame

    :return: DataFrame with columns:
        ["rid", "iti_id", "path_id", "seg_id", "alight_ts", "board_ts", "transfer_time"]
        
        - `seg_id`: the alighted train segment id
        - `transfer_time`: the time difference between the `alight_ts` of `seg_id` and the `board_ts` of `seg_id` + 1.
    :rtype: pd.DataFrame
    """
    # delete (rid, iti_id) with only one seg
    seg_count = df_feas_iti.groupby(['rid', 'iti_id'])[
        'seg_id'].transform('nunique')
    df = df_feas_iti.loc[seg_count > 1].copy()

    # find rows that need to combine next row's board_ts
    df['next_index'] = df.groupby(
        ['rid', 'iti_id'])['seg_id'].shift(-1).notna()

    # calculate transfer time
    df["next_board_ts"] = df["board_ts"].shift(-1)
    df = df[df["next_index"]]
    df["next_board_ts"] = df["next_board_ts"].astype(int)
    df["transfer_time"] = df["next_board_ts"] - df["alight_ts"]

    # get essential data
    df.drop(columns=["board_ts", "train_id", "next_index"], inplace=True)
    df.rename(columns={"next_board_ts": "board_ts"}, inplace=True)

    return df


def get_transfer_from_assigned() -> pd.DataFrame:
    """
    Find transfer times from assigned_*.pkl files.

    :return: DataFrame with columns:
        ["rid", "path_id", "seg_id", "alight_ts", "transfer_time"]
        
        - `seg_id`: the alighted train segment id
    :rtype: pd.DataFrame
    
    """
    assigned = read_all(config.CONFIG["results"]["assigned"], show_timer=False)

    return get_transfer_from_feas_iti(assigned).drop(columns=["board_ts", "iti_id"])


def get_path_seg_to_pp_ids() -> pd.DataFrame:
    """
    Get path_id, seg_id to physical platform IDs mapping dataframe.

    :return: DataFrame with 5 columns:
        ["path_id", "seg_id", "transfer_type", "pp_id1", "pp_id2"]

        - `pp_id1`: the physical platform ID of the 'from' node of the transfer link
        - `pp_id2`: the physical platform ID of the 'to' node of the transfer link
    :rtype: pd.DataFrame
    """
    k_pv_ = get_k_pv()[:, :-2]
    df = pd.DataFrame(
        k_pv_, columns=["path_id", "pv_id", "node1", "node2", "link_type"])

    # delete non-transfer pathvia rows
    path_id, seg_count = np.unique(df["path_id"], return_counts=True)
    path_id_with_transfer = path_id[seg_count > 3]
    df = df[df['path_id'].isin(path_id_with_transfer)]

    # add seg_id for in_vehicle link
    in_vehicle_df = df[df['link_type'] == 'in_vehicle'].copy()
    in_vehicle_df['seg_id'] = in_vehicle_df.groupby('path_id').cumcount() + 1
    df = pd.merge(df, in_vehicle_df[['seg_id']],
                  left_index=True, right_index=True, how='left')

    # delete first and last pathvia row (entry, egress)
    first_pv_ind = (df["pv_id"] == 1)  # entry
    last_pv_ind = first_pv_ind.shift(-1)  # egress
    df = df[~(first_pv_ind | last_pv_ind)].iloc[:-1, :]

    # aggregate egress-entry transfer link
    df["seg_id"] = df["seg_id"].ffill().astype(int)
    df["next_node2"] = df["node2"].shift(-1)
    # fix swap expressions
    df.loc[df["link_type"] == "platform_swap", "next_node2"] = df["node2"]
    df.loc[df["link_type"] == "egress",
           "link_type"] = "egress-entry"  # rename for clarity
    mapper_ps2nodes = df[df["link_type"].isin(["egress-entry", "platform_swap"])][
        ["path_id", "seg_id", "node1", "next_node2", "link_type"]].rename(
        columns={"next_node2": "node2", "link_type": "transfer_type"}
    )

    # map 2 node_id to 2 physical_platform_id (pp_id)
    platform = pd.DataFrame(get_platform(
    )[:, :-1], columns=["physical_platform_id", "node_id"]).set_index("node_id")

    mapper_ps2nodes["pp_id1"] = mapper_ps2nodes["node1"].map(
        platform["physical_platform_id"])
    mapper_ps2nodes["pp_id2"] = mapper_ps2nodes["node2"].map(
        platform["physical_platform_id"])
    mapper_ps2nodes.drop(columns=["node1", "node2"], inplace=True)
    return mapper_ps2nodes


def filter_transfer_all() -> pd.DataFrame:
    """
    Find transfer times from assigned_*.pkl files, map them with physical_platform_id.

    :return: DataFrame with columns:
        ["rid" (index), "path_id", "seg_id", "pp_id1", "pp_id2", "alight_ts", "transfer_time", "transfer_type"]

        - `pp_id1`: the physical platform ID of the 'from' node of the transfer link
        - `pp_id2`: the physical platform ID of the 'to' node of the transfer link
        - `transfer_type`: the type of the transfer link, either "egress-entry" or "platform_swap"
    :rtype: pd.DataFrame
    """
    df_tt = get_transfer_from_assigned(
    )  # [rid, path_id, seg_id, alight_ts, transfer_time]
    # [path_id, seg_id, pp_id1, pp_id2, transfer_type]
    df_mapper = get_path_seg_to_pp_ids()
    # [rid, path_id, seg_id, alight_ts, transfer_time, pp_id1, pp_id2, transfer_type]
    df = df_tt.merge(df_mapper, on=["path_id", "seg_id"], how="left")
    assert df[df["pp_id1"].isna() | df["pp_id2"].isna()
              ].shape[0] == 0, "Transfer pp_id not found!"
    df = df[["rid", "path_id", "seg_id", "pp_id1", "pp_id2",
             "alight_ts", "transfer_time", "transfer_type"]].set_index("rid")
    df["transfer_time"] = df["transfer_time"].astype(int)
    df["transfer_type"] = df["transfer_type"].astype("category")
    return df


if __name__ == "__main__":
    config.load_config()
    
    df = filter_transfer_all()
    print(df.shape)
    print(df.head())