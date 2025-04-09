# to test and implement GPT-generated code
import pandas as pd

from src.utils import read_data, file_saver


def find_rids_with_same_final_train_in_all_itis():
    """
    Find rids in feas_iti_left.pkl where all feasible itineraries share the same final train (same train_id & alight_ts).

    Returns
    -------
    pd.DataFrame
        Subset of feas_iti_left containing only those rids whose all iti end with the same train.
    """
    df = read_data("feas_iti_left")

    # Keep only the last segment for each rid + iti_id
    # last_seg = df.sort_values("seg_id").groupby(["rid", "iti_id"]).tail(1)
    last_seg = df.groupby(["rid", "iti_id"]).last().reset_index()

    # 对每个 rid，聚合 (train_id, alight_ts) 为元组，统计唯一值数量
    last_seg["train_end"] = list(zip(last_seg["train_id"], last_seg["alight_ts"]))
    unique_end_count = last_seg.groupby("rid")["train_end"].nunique()

    # 筛选所有 iti 最终列车一致的 rid
    consistent_rids = unique_end_count[unique_end_count == 1].index

    return df[df["rid"].isin(consistent_rids)]


if __name__ == '__main__':
    # df = find_rids_with_same_final_train_in_all_itis()
    # print(df.sample(n=20))
    # from scripts.find_feas_iti import _plot_check_feas_iti
    # _plot_check_feas_iti(rid=347627)
    KPV = read_data("pathvia")
    print(KPV[KPV['path_id'].isin([1136109701, 1136109702, 1136109703])])
    pass
