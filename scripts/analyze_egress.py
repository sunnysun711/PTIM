import numpy as np
import pandas as pd

from src import config
from src.utils import save_
from src.walk_time_dis import get_egress_time_from_feas_iti_left, get_egress_time_from_feas_iti_assigned, \
    plot_egress_time_dis_all


def _check_egress_time_outlier_rejects(uid: int = 1124):
    config.load_config()
    from src.utils import read_
    from src.walk_time_dis import get_egress_link_groups, get_reject_outlier_bd, plot_egress_time_dis

    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False)
    et__ = et_.set_index(["node1", "node2"])

    # uid = 1018  # Tianfu 3th Road
    # uid = 1064  # Tianfu 5th Road
    # uid = 1123  # GuangDu
    # uid = 1124  # XiBoCheng
    # uid = 1060  # ShuangLiu T1
    platform_links = get_egress_link_groups(et_=et_)[uid]

    print(uid, platform_links)
    for egress_links in platform_links:
        # all egress links on the same platform
        et = et__[et__.index.isin(egress_links)]
        if et.shape[0] == 0:
            continue
        print("Data size: ", et.shape[0], egress_links)
        plot_egress_time_dis(
            egress_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=f"{egress_links}",
            show_=True  # save
        )

        lb, ub = get_reject_outlier_bd(data=et['egress_time'].values, method="zscore", abs_max=None)
        et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
        print(f"Bounded by [{lb}, {ub}]: ", et.shape[0])

        plot_egress_time_dis(
            egress_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=f"{egress_links}",
            show_=True  # save
        )
    return


def save_egress_times(save_on: bool = False) -> pd.DataFrame:
    """
    Save a combined DataFrame of egress times from left.pkl and assigned_1.pkl.
    "rid" as index.
    ["node1", "node2", "alight_ts", "ts2", "egress_time"] as columns.
    """
    df1 = get_egress_time_from_feas_iti_left()
    df2 = get_egress_time_from_feas_iti_assigned()
    df = pd.concat([df1, df2], ignore_index=True)
    if save_on:
        save_(fn=config.CONFIG["results"]["egress_times"], data=df, auto_index_on=True)
    return df


def plot_egress_times(save_subfolder: str = "ETD0", et_: pd.DataFrame = None):
    plot_egress_time_dis_all(save_subfolder=save_subfolder, et_=et_, save_on=True)
    pass


def main():
    info_message = (
        "\033[33m"
        "======================================================================================\n"
        "[INFO] This script performs the following operations:\n"
        "       1. Generate egress time data file in `egress_times_1.pkl`.\n"
        "          - index: \"rid\".\n"
        "          - columns: [\"node1\", \"node2\", \"alight_ts\", \"ts2\", \"egress_time\"].\n"
        "       2. Generate egress time distribution plots in `results/ETD0`.\n"
        "======================================================================================\n"
        "\033[0m"
    )
    print(info_message)
    et_ = save_egress_times(save_on=True)
    plot_egress_times(save_subfolder="ETD0", et_=et_)
    pass


if __name__ == '__main__':
    # main()
    # _check_egress_time_outlier_rejects(uid=1024)
    pass
