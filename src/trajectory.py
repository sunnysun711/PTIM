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
- assigned_1.pkl (will generate)
- left.pkl (will generate)
"""
import os

import pandas as pd

from src import config
from src.utils import read_, save_


def split_feas_iti(feas_iti_cnt_limit: int = None):
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
        Default is given by config yaml file.
    :return:
    """
    feas_iti_cnt_limit = config.CONFIG["parameters"][
        "feas_iti_cnt_limit"] if feas_iti_cnt_limit is None else feas_iti_cnt_limit
    FI = read_(fn=config.CONFIG["results"]["feas_iti"], show_timer=True)  # takes 1 second to load data
    df = FI.drop_duplicates(["rid"], keep="last")

    rid_to_assign_list = df[df['iti_id'] == 1].rid.tolist()
    rid_to_stash_list = df[df['iti_id'] > feas_iti_cnt_limit].rid.tolist()
    rid_to_left_list = df[~df['rid'].isin(rid_to_assign_list + rid_to_stash_list)].rid.tolist()

    # Filter the main DataFrame based on the above lists
    assigned_df = FI[FI['rid'].isin(rid_to_assign_list)]
    stashed_df = FI[FI['rid'].isin(rid_to_stash_list)]
    left_df = FI[FI['rid'].isin(rid_to_left_list)]

    # Save the filtered DataFrames to new files
    save_(fn=config.CONFIG["results"]["assigned"], data=assigned_df, auto_index_on=True)
    save_(fn=config.CONFIG["results"]["stashed"], data=stashed_df, auto_index_on=False)
    save_(fn=config.CONFIG["results"]["left"], data=left_df, auto_index_on=False)

    print(f"[INFO] Split {config.CONFIG['results']['feas_iti']} into three files:")
    print(f"[INFO] 1. assigned_1.pkl: {len(assigned_df)} rows")
    print(f"[INFO] 2. stashed.pkl: {len(stashed_df)} rows")
    print(f"[INFO] 3. left.pkl: {len(left_df)} rows")
    return


def assign_feas_iti_to_trajectory(rid_iti_id_pair: list[tuple[int, int]]):
    """
    Assign feasible itineraries to trajectories.
    """
    fi_left = read_(config.CONFIG["results"]['left'], show_timer=False)

    assigned_df = fi_left.merge(
        pd.DataFrame(rid_iti_id_pair, columns=["rid", "iti_id"]),
        on=["rid", "iti_id"],
        how="inner"
    )

    # save feas_iti_assigned
    save_(fn=config.CONFIG["results"]["assigned"], data=assigned_df, auto_index_on=True)

    # override and save feas_iti_left
    remaining_df = fi_left[~fi_left["rid"].isin(assigned_df["rid"])]
    save_(fn=config.CONFIG["results"]["left"], data=remaining_df, auto_index_on=False)
    return


def roll_back_assignment():
    """
    Roll back the latest feasible itinerary assignment.
    """
    from src.utils import get_file_path, get_latest_file_index
    fp = get_file_path(config.CONFIG["results"]["assigned"])
    latest_version = get_latest_file_index(fp, get_next=False)
    fp = fp.split(".")[0] + f"_{latest_version}." + fp.split(".")[-1]

    assigned_df = read_(config.CONFIG["results"]["assigned"], latest_=True)
    rid_list = assigned_df["rid"].unique().tolist()
    print("rid_list", rid_list)

    os.remove(fp)
    print(f"[INFO] Removed {fp}.")

    # find assigned rid in feas_iti.pkl
    fi_all = read_(config.CONFIG["results"]["feas_iti"], show_timer=False)
    to_restore = fi_all[fi_all["rid"].isin(rid_list)]
    print("to_restore shape", to_restore.shape)

    # add related info back to feas_iti_left.pkl
    fi_left = read_(config.CONFIG["results"]["left"], show_timer=False)
    fi_left = pd.concat([fi_left, to_restore])
    save_(fn=config.CONFIG["results"]["left"], data=fi_left, auto_index_on=False)

    return


def main():
    ...


if __name__ == '__main__':
    main()
    pass
