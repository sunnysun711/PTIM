import numpy as np
import pandas as pd

from src import config
from src.utils import save_, read_
from src.walk_time_dis import *


def _check_transfer_links_distribution():
    pass


def save_transfer_times(save_on: bool = False) -> pd.DataFrame:
    """
    Save the transfer times filtered from assigned trajectories to a PKL file.
    :return: DataFrame with 4 columns:
        ["path_id", "seg_id", "alight_ts", "transfer_time"]
        where seg_id is the alighting train segment id, the transfer time is thus considered as the time difference
        between the alighting time of seg_id and the boarding time of seg_id + 1.
    """
    df = filter_transfer_time_from_assigned()
    if save_on:
        save_(fn=config.CONFIG["results"]["transfer_times"], data=df, auto_index_on=True)
    return df


def save_transfer_links_info(save_on: bool = False) -> pd.DataFrame:
    """
    Save the transfer links information to a CSV file.
    :return: A DataFrame with 7 columns:
        ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"]
        where seg_id is the alighting train segment id, transfer_type is one of "platform_swap", "egress-entry",
        and p_uid_min, p_uid_max are the platform_uids of the two platforms involved in the transfer.
    """
    df = map_path_seg_to_transfer_link()
    if save_on:
        save_(fn=config.CONFIG["results"]["transfer_links"], data=df, auto_index_on=True)
    return df


def save_ttd(df_tt, map_p2t, save_on: bool = False) -> pd.DataFrame:
    """
    Save the transfer time distribution to a CSV file.
    :return: A DataFrame with 5 columns:
        ["path_id", "seg_id", "node1", "node2", "transfer_time", "pdf", "cdf"]
        where seg_id is the alighting train segment id, transfer_time is the time difference between the alighting time
        of seg_id and the boarding time of seg_id + 1.
    """
    df = fit_transfer_time_dis_all(df_tt=df_tt, map_p2t=map_p2t)
    if save_on:
        save_(fn=config.CONFIG["results"]["ttd"], data=df, auto_index_on=True)
    return df


def main():
    df_tt = save_transfer_times(save_on=True)
    map_p2t = save_transfer_links_info(save_on=True)
    save_ttd(df_tt, map_p2t, save_on=True)
    ...


if __name__ == '__main__':
    config.load_config()
    main()
