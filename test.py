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
    last_seg = df.sort_values("seg_id").groupby(["rid", "iti_id"]).tail(1)

    # For each rid, check if all its iti share the same (train_id, alight_ts)
    rid_grp = last_seg.groupby("rid")[["train_id", "alight_ts"]].agg(["nunique", "first"])
    consistent_rids = rid_grp[
        (rid_grp[("train_id", "nunique")] == 1) &
        (rid_grp[("alight_ts", "nunique")] == 1)
        ].index

    return df[df["rid"].isin(consistent_rids)]



if __name__ == '__main__':
    # df = find_rids_with_same_final_train_in_all_itis()
    # df = assign_feas_iti_to_trajectory([(13541, 2), (142352, 1)], save_fn="haha")
    # print(df)
    from src.trajectory import split_feas_iti, assign_feas_iti_to_trajectory, roll_back_assignment

    # split_feas_iti()
    # assign_feas_iti_to_trajectory([(13541, 2), (142352, 1)])
    roll_back_assignment()
    pass
