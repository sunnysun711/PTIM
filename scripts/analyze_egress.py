import numpy as np
import pandas as pd

from src import config
from src.utils import save_, read_
from src.walk_time_dis import get_egress_time_from_left, get_egress_time_from_assigned, \
    plot_egress_time_dis_all, get_physical_links_info, fit_egress_time_dis_all_parallel


def _check_egress_time_outlier_rejects(uid: int = None):
    config.load_config()
    from src.utils import read_
    from src.walk_time_dis import get_reject_outlier_bd, plot_egress_time_dis

    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False)

    # uid = uid if uid else np.random.choice(range(1001, 1137), size=1)[0]
    uid = 1032
    # uid = 1018  # Tianfu 3th Road
    # uid = 1064  # Tianfu 5th Road
    # uid = 1123  # GuangDu
    # uid = 1124  # XiBoCheng
    # uid = 1060  # ShuangLiu T1
    pl_info = get_physical_links_info(platform=None, et_=et_)
    pl_info = pl_info[pl_info[:, 2] == uid]

    print(uid, pl_info)
    pl_ids = np.unique(pl_info[:, 0])

    for pl_id in pl_ids:
        print(pl_id)
        platform_ids = pl_info[pl_info[:, 0] == pl_id, 1]
        et = et_[et_["node1"].isin(platform_ids)]
        if et.shape[0] == 0:
            continue

        print("Data size: ", et.shape[0], platform_ids)

        plot_egress_time_dis(
            egress_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=f"All data {platform_ids}",
            show_=True  # save if show_ is False
        )

        lb, ub = get_reject_outlier_bd(data=et['egress_time'].values, method="zscore", abs_max=None)
        et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
        print(f"Bounded by [{lb}, {ub}]: ", et.shape[0])
        plot_egress_time_dis(
            egress_time=et['egress_time'].values, alight_ts=et['alight_ts'].values, title=f"Bounded {platform_ids}",
            show_=True  # save if show_ is False
        )
    return


def save_egress_times(save_on: bool = False) -> pd.DataFrame:
    """
    Save a combined DataFrame of egress times from left.pkl and assigned_1.pkl.
    "rid" as index.
    ["node1", "node2", "alight_ts", "ts2", "egress_time"] as columns.
    """
    df1 = get_egress_time_from_left()
    df2 = get_egress_time_from_assigned()
    df = pd.concat([df1, df2], ignore_index=True)
    if save_on:
        save_(fn=config.CONFIG["results"]["egress_times"], data=df, auto_index_on=True)
    return df


def save_physical_links_info(et_: pd.DataFrame, save_on: bool = False) -> pd.DataFrame:
    """
    Save the physical links information to a CSV file.
    """
    pl_info = get_physical_links_info(platform=None, et_=et_)
    df_pl = pd.DataFrame(pl_info, columns=["pl_id", "platform_id", "uid"]).set_index("pl_id")
    if save_on:
        save_(fn=config.CONFIG["results"]["physical_links"], data=df_pl, auto_index_on=True)
    return df_pl


def read_physical_links_info(index: int = None) -> pd.DataFrame:
    from src.utils import split_fn_index_ext
    full_fn = config.CONFIG["results"]["physical_links"]
    fn, _, ext = split_fn_index_ext(full_fn)
    if index is None:
        fn = fn + ext
        return read_(fn=fn, show_timer=True, latest_=True)
    fn = f"{fn}_{index}{ext}"
    df = read_(fn=fn, show_timer=True)
    return df


def save_etd(et_: pd.DataFrame, save_on: bool = False) -> pd.DataFrame:
    etd = fit_egress_time_dis_all_parallel(et_=et_)
    if save_on:
        save_(fn=config.CONFIG["results"]["etd"], data=etd, auto_index_on=True)
    return etd


def main(physical_links_index: int = None):
    info_message = (
        "\033[33m"
        "======================================================================================\n"
        "[INFO] This script performs the following operations:\n"
        "       1. Generate egress time data file in `egress_times_1.pkl`.\n"
        "          - index: \"rid\".\n"
        "          - columns: [\"node1\", \"node2\", \"alight_ts\", \"ts2\", \"egress_time\"].\n"
        "       2. Generate egress time distribution plots in `figures/ETD0/`.\n"
        "       3. Save physical links information to `results/egress/physical_links_1.csv`.\n"
        "          - columns: [\"pl_id\", \"platform_id\", \"uid\"].\n"
        "       4. Save ETD (Egress Time Distribution) results to `results/egress/etd_1.pkl`.\n"
        "          - columns: [\"pl_id\", \"x\", \n"
        "                      \"kde_pdf\", \"kde_cdf\", \"kde_ks_stat\", \"kde_ks_p_value\", \n"
        "                      \"gamma_pdf\", \"gamma_cdf\", \"gamma_ks_stat\", \"gamma_ks_p_value\", \n"
        "                      \"lognorm_pdf\", \"lognorm_cdf\", \"lognorm_ks_stat\", \"lognorm_ks_p_value\" \n"
        "                     ]\n"
        "======================================================================================\n"
        "\033[0m"
    )
    print(info_message)
    et_ = save_egress_times(save_on=True)

    if physical_links_index is None:
        df_pl = save_physical_links_info(et_=et_, save_on=True)
    else:
        df_pl = read_physical_links_info(index=physical_links_index)

    plot_egress_time_dis_all(save_subfolder="ETD0", et_=et_, physical_link_info=df_pl.values, save_on=True)
    save_etd(et_=et_, save_on=True)
    pass


if __name__ == '__main__':
    config.load_config()
    # main()
    # _check_egress_time_outlier_rejects(uid=None)
    pass
