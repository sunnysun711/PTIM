import numpy as np
import pandas as pd

from src import config
from src.utils import save_, read_
from src.walk_time_dis import *


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
        save_(fn=config.CONFIG["results"]["transfer_links"], data=df.set_index("path_id"), auto_index_on=True)
    return df


def read_transfer_links_info(index: int = None) -> pd.DataFrame:
    """
    :param index:
    :return: DataFrame with columns: ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"].
    """
    from src.utils import split_fn_index_ext
    full_fn = config.CONFIG["results"]["transfer_links"]
    fn, _, ext = split_fn_index_ext(full_fn)
    if index is None:
        return read_(fn=fn + ext, show_timer=True, latest_=True)
    return read_(fn=f"{fn}_{index}{ext}", show_timer=True)


def save_ttd(df_tt, map_p2t, save_on: bool = False) -> pd.DataFrame:
    """
    Save the transfer time distribution to a CSV file.
    :return: A DataFrame with 6 columns:
        [p_uid1, p_uid2, x, kde_cdf, gamma_cdf, lognorm_cdf]
        where p_uid1, p_uid2 are smaller and larger platform_uid of the transfer link.
    """
    df = fit_transfer_time_dis_all(df_tt=df_tt, map_p2t=map_p2t)
    if save_on:
        save_(fn=config.CONFIG["results"]["ttd"], data=df, auto_index_on=True)
    return df


def plot_ttd(df_tt, map_p2t):
    """
    Plot the transfer time distribution.
    :param df_tt: DataFrame with 4 columns:
        ["path_id", "seg_id", "alight_ts", "transfer_time"]
        where seg_id is the alighting train segment id, the transfer time is thus considered as the time difference
        between the alighting time of seg_id and the boarding time of seg_id + 1.
    :param map_p2t: A DataFrame with 7 columns:
        ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"]
        where seg_id is the alighting train segment id, transfer_type is one of "platform_swap", "egress-entry",
        and p_uid_min, p_uid_max are the platform_uids of the two platforms involved in the transfer.
    """
    plot_transfer_time_dis_all(df_tt=df_tt, map_p2t=map_p2t, save_subfolder="TTD0", save_on=True)
    pass


def main(use_transfer_links_index: int = None):
    """
    Main function to run the analysis.
    :param use_transfer_links_index: Index of the transfer links file to use. If None, use the latest one.
    """
    info_message = (
        "\033[33m"
        "======================================================================================\n"
        "[INFO] This script performs the following operations:\n"
        "       1. Generate transfer time data file in `transfer_times_1.pkl`.\n"
        "          - columns: [\"path_id\", \"seg_id\", \"alight_ts\", \"transfer_time\"].\n"
        "       2. Generate transfer time distribution plots in `figures/TTD0/`.\n"
        "       3. Save transfer links information to `results/transfer/transfer_links_1.csv`.\n"
        "          - columns: [\"path_id\", \"seg_id\", \"node1\", \"node2\", \"p_uid_min\", \n"
        "                      \"p_uid_max\", \"transfer_type\"].\n"
        "       4. Save TTD (Transfer Time Distribution) results to `results/transfer/ttd_1.pkl`.\n"
        "          - columns: [\"p_uid1\", \"p_uid2\", \"x\", \"kde_cdf\", \"gamma_cdf\", \"lognorm_cdf\"] \n"
        "======================================================================================\n"
        "\033[0m"
    )
    print(info_message)

    df_tt = save_transfer_times(save_on=True)
    map_p2t = save_transfer_links_info(save_on=True) if use_transfer_links_index is None else \
        read_transfer_links_info(index=use_transfer_links_index)
    plot_ttd(df_tt, map_p2t)
    save_ttd(df_tt, map_p2t, save_on=True)
    ...


if __name__ == '__main__':
    config.load_config()
    main()
