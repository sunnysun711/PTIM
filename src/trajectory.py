"""
This module manages feasible itinerary assignment and partitioning for passenger trajectories.

Core Responsibilities:
1. Select and assign valid itineraries from feas_iti_left.pkl based on rules (e.g., probabilities in feas_iti_prob.pkl).
2. Save newly assigned itineraries into feas_iti_assigned_X.pkl (via file_auto_index_saver).
3. Optionally split feas_iti.pkl into three categories for initial preprocessing:
    - feas_iti_assigned.pkl: Only one itinerary per rid
    - feas_iti_stashed.pkl: Too many itineraries (above threshold)
    - feas_iti_left.pkl: Multiple but manageable itineraries (to-be-assigned)

This module is crucial for iterative, rule-based refinement and reassignment of feasible itineraries.

Key Functions:
- split_feas_iti: One-time utility to prepare initial data subsets
- (To be implemented): rule_based_assignment_from_left

Dependencies:
- src.utils: File I/O handling

Data Sources:
- feas_iti.pkl
- feas_iti_assigned.pkl (will generate)
- feas_iti_left.pkl (will generate)
"""
import os

import pandas as pd

from src.utils import read_data, file_auto_index_saver, file_saver, read_data_latest
from src.utils import find_data_latest


def split_feas_iti(feas_iti_cnt_limit: int = 1000):
    """
    This is a one-step method that splits the feas_iti.pkl into three files:
        1. feas_iti_assigned_1.pkl:
            only one feasible itinerary for each rid.
        2. feas_iti_stashed.pkl:
            more than `feas_iti_cnt_limit` feasible itineraries for each rid.
        3. feas_iti_left.pkl:
            less than `feas_iti_cnt_limit` but more than 1 feasible itineraries for each rid.
    :param feas_iti_cnt_limit:
        The maximum number of feasible itineraries for each rid.
        If the number of feasible itineraries for a rid is greater than this limit,
        the corresponding row will be saved to feas_iti_stashed.pkl.
        Default is 1000.
    :return:
    """
    FI = read_data("feas_iti", show_timer=True)  # takes 1 second to load data
    df = FI.drop_duplicates(["rid"], keep="last")

    rid_to_assign_list = df[df['iti_id'] == 1].rid.tolist()
    rid_to_stash_list = df[df['iti_id'] > feas_iti_cnt_limit].rid.tolist()
    rid_to_left_list = df[~df['rid'].isin(rid_to_assign_list + rid_to_stash_list)].rid.tolist()

    # Filter the main DataFrame based on the above lists
    assigned_df = FI[FI['rid'].isin(rid_to_assign_list)]
    stashed_df = FI[FI['rid'].isin(rid_to_stash_list)]
    left_df = FI[FI['rid'].isin(rid_to_left_list)]

    # Save the filtered DataFrames to new files
    file_auto_index_saver(lambda save_fn: assigned_df)(save_fn="feas_iti_assigned")
    file_saver(lambda save_fn: stashed_df)(save_fn="feas_iti_stashed")
    file_saver(lambda save_fn: left_df)(save_fn="feas_iti_left")

    print(f"[INFO] Split feas_iti.pkl into three files:")
    print(f"[INFO] 1. feas_iti_assigned_1.pkl: {len(assigned_df)} rows")
    print(f"[INFO] 2. feas_iti_stashed.pkl: {len(stashed_df)} rows")
    print(f"[INFO] 3. feas_iti_left.pkl: {len(left_df)} rows")
    return


def assign_feas_iti_to_trajectory(rid_iti_id_pair: list[tuple[int, int]]):
    """
    Assign feasible itineraries to trajectories.
    """
    fi_left = read_data("feas_iti_left", show_timer=False)

    assigned_df = fi_left.merge(
        pd.DataFrame(rid_iti_id_pair, columns=["rid", "iti_id"]),
        on=["rid", "iti_id"],
        how="inner"
    )

    # save feas_iti_assigned
    file_auto_index_saver(lambda save_fn: assigned_df)(save_fn="feas_iti_assigned")

    # override and save feas_iti_left
    remaining_df = fi_left[~fi_left["rid"].isin(assigned_df["rid"])]
    file_saver(lambda save_fn: remaining_df)(save_fn="feas_iti_left")

    return


def roll_back_assignment():
    """
    Roll back the latest feasible itinerary assignment.
    """
    file_latest = find_data_latest("feas_iti_assigned")
    assigned_df = pd.read_pickle(file_latest)
    rid_list = assigned_df["rid"].unique().tolist()
    print("rid_list", rid_list)

    os.remove(file_latest)
    print(f"[INFO] Removed {file_latest}.")

    # find assigned rid in feas_iti.pkl
    fi_all = read_data("feas_iti", show_timer=False)
    to_restore = fi_all[fi_all["rid"].isin(rid_list)]
    print("to_restore shape", to_restore.shape)

    # add related info back to feas_iti_left.pkl
    fi_left = read_data("feas_iti_left", show_timer=False)
    fi_left = pd.concat([fi_left, to_restore])
    file_saver(lambda save_fn: fi_left)(save_fn="feas_iti_left")
    return


def main():
    ...


if __name__ == '__main__':
    main()
    pass
