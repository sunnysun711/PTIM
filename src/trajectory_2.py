"""
This module manages feasible itinerary assignment and partitioning for passenger trajectories.

Core Responsibilities
---------------------
1. Select and assign valid itineraries from ``feas_iti_left.pkl`` based on rules (e.g., probabilities in ``feas_iti_prob.pkl``).
2. Save newly assigned itineraries into ``feas_iti_assigned_X.pkl`` (auto-incremented filenames).
3. Optionally split ``feas_iti.pkl`` into three categories for preprocessing:

   - ``feas_iti_assigned.pkl``: Only one itinerary per rid
   - ``feas_iti_stashed.pkl``: Too many itineraries (above threshold)
   - ``feas_iti_left.pkl``: Multiple but manageable itineraries (to-be-assigned)

This module is crucial for iterative, rule-based refinement and reassignment of feasible itineraries.

Key Functions
-------------
- ``split_feas_iti``: One-time utility to prepare initial data subsets
- ``dynamic_assignment``: Iteratively assign feasible itineraries with overload handling

Dependencies
------------
- src.utils: File I/O handling
- src.itinerary: Itinerary probability and penalty computation
- src.timetable: Overload train section detection

"""
import os

import numpy as np
import pandas as pd

from src import config
from src.congest_penal import build_penal_mapper_df
from src.itinerary import cal_in_vehicle_penal_all, compute_itinerary_probabilities, filter_dis_file
from src.timetable import find_overload_train_section
from src.utils import read_, read_all, save_


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
    :type feas_iti_cnt_limit: int, optional
    """
    feas_iti_cnt_limit = config.CONFIG["parameters"][
        "feas_iti_cnt_limit"] if feas_iti_cnt_limit is None else feas_iti_cnt_limit
    # takes 1 second to load data
    FI = read_(fn=config.CONFIG["results"]["feas_iti"], show_timer=True)
    df = FI.drop_duplicates(["rid"], keep="last")

    rid_to_assign_list = df[df['iti_id'] == 1].rid.tolist()
    rid_to_stash_list = df[df['iti_id'] > feas_iti_cnt_limit].rid.tolist()
    rid_to_left_list = df[~df['rid'].isin(
        rid_to_assign_list + rid_to_stash_list)].rid.tolist()

    # Filter the main DataFrame based on the above lists
    assigned_df = FI[FI['rid'].isin(rid_to_assign_list)]
    stashed_df = FI[FI['rid'].isin(rid_to_stash_list)]
    left_df = FI[FI['rid'].isin(rid_to_left_list)]

    # Save the filtered DataFrames to new files
    save_(fn=config.CONFIG["results"]["assigned"],
          data=assigned_df, auto_index_on=True)
    save_(fn=config.CONFIG["results"]["stashed"],
          data=stashed_df, auto_index_on=False)
    save_(fn=config.CONFIG["results"]["left"],
          data=left_df, auto_index_on=False)

    print(
        f"[INFO] Split {config.CONFIG['results']['feas_iti']} into three files:")
    print(f"[INFO] 1. assigned_1.pkl: {len(assigned_df)} rows")
    print(f"[INFO] 2. stashed.pkl: {len(stashed_df)} rows")
    print(f"[INFO] 3. left.pkl: {len(left_df)} rows")
    return


def _get_sorted_data_and_first_idx_and_uniqueness(feas_iti_prob: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal helper: return sorted data, unique_rids, first_idx, is_unique_max.
    Use numpy lexsort to enhance performance.
    
    :param feas_iti_prob: DataFrame with index [rid, iti_id], column 'prob'.
    :return: tuple of (sorted_data, unique_rids, first_idx, is_unique_max).
    """
    data = feas_iti_prob.reset_index()[["rid", "iti_id", "prob"]].to_numpy()
    sorted_idx = np.lexsort((-data[:, 2], data[:, 0]))
    data_sorted = data[sorted_idx]
    unique_rids, first_idx = np.unique(data_sorted[:, 0], return_index=True)

    prob_first = data_sorted[first_idx, 2]
    # handle boundary: avoid index error if last index
    prob_next = np.zeros_like(prob_first)
    prob_next[:-1] = data_sorted[first_idx[1:], 2]
    prob_next[-1] = -np.inf  # last rid no next, force unique

    is_unique_max = prob_first != prob_next

    return data_sorted, unique_rids, first_idx, is_unique_max


def filter_most_probable_iti(
    feas_iti_prob: pd.DataFrame,
    sort_prob: bool = True,
    exclude_non_unique_max_prob: bool = False
) -> pd.DataFrame:
    """
    Filter the most probable itinerary for each rid.

    :param feas_iti_prob:
        DataFrame with at least: ["rid" (index), "iti_id" (index), "prob"]

    :param sort_prob:
        Whether to sort the result by prob descending. Default True.

    :param exclude_non_unique_max_prob:
        If True, exclude rids that have multiple itineraries sharing the same max prob.

    :return:
        DataFrame with columns ["rid", "iti_id", "prob"].
    """
    data_sorted, unique_rids, first_idx, is_unique_max = _get_sorted_data_and_first_idx_and_uniqueness(
        feas_iti_prob)

    # filter based on is_unique_max
    if exclude_non_unique_max_prob:
        valid_idx = first_idx[is_unique_max]
    else:
        valid_idx = first_idx

    data_result = data_sorted[valid_idx]
    most_probable_iti_df = pd.DataFrame(
        data_result, columns=["rid", "iti_id", "prob"])
    most_probable_iti_df['rid'] = most_probable_iti_df['rid'].astype(int)
    most_probable_iti_df['iti_id'] = most_probable_iti_df['iti_id'].astype(int)

    if sort_prob:
        most_probable_iti_df.sort_values("prob", ascending=False, inplace=True)
        most_probable_iti_df.reset_index(drop=True, inplace=True)

    return most_probable_iti_df


def find_multiple_max_prob_rids(feas_iti_prob: pd.DataFrame) -> np.ndarray:
    """
    Find rid with multiple max-prob itineraries. Indicating that these rids may
    have multiple itineraries that share a same egress time and all above 500s
    entry and transfer links, which shares the total 100% prob.

    :param feas_iti_prob:
        The probability of each itinerary for each rid.

        Expected columns (at least): ["rid" (index), "iti_id" (index), "prob"]
        Should generated from `src.itinerary.compute_itinerary_probabilities()`
    :type feas_iti_prob: pd.DataFrame

    :return: Array of non_unique_max_prob_itinerary_rids.
    :rtype: np.ndarray
    """
    data_sorted, unique_rids, first_idx, is_unique_max = _get_sorted_data_and_first_idx_and_uniqueness(
        feas_iti_prob)

    # prob_first = data_sorted[first_idx, 2]
    # # second max prob, potentially equal to max_prob
    # prob_next = data_sorted[first_idx + 1, 2]

    # is_unique_max = prob_first != prob_next  # check uniqueness

    non_unique_rids = unique_rids[~is_unique_max]
    # print(f"Number of rid with non-unique max prob: {len(non_unique_rids)}")
    # print(f"Example rids: {np.random.choice(non_unique_rids, size=10)}")

    # flag_df = pd.DataFrame({
    #     'rid': unique_rids,
    #     'is_unique_max': is_unique_max,
    #     'prob': prob_first
    # })
    # flag_df = flag_df[~is_unique_max]
    # return flag_df

    return non_unique_rids


def assign_feas_iti_to_trajectory(rid_iti_pairs: np.ndarray | list[tuple[int, int]]):
    """
    Assign feasible itineraries to trajectories by moving selected records from
    the left feasible itinerary file to the assigned file, and updating the left file.

    This function performs the following steps:

    1. Reads the current `left.pkl` file (specified in config) containing all 
       remaining feasible itineraries.
    2. Filters the rows whose [rid, iti_id] pairs match the given `rid_iti_pairs`.
    3. Saves the matched rows to the `assigned_*.pkl` file (specified in config).
       If the file already exists, appends a new version with an incremental index.
    4. Updates `left.pkl` by removing the assigned `rid` rows and overwriting the file.

    Each time this function is called, it moves a batch of assigned itineraries
    from the left pool to the assigned pool, progressively reducing the left file.

    --------
    File operations:

    - Reads: config.CONFIG["results"]['left'] → left.pkl
    - Writes:
        - config.CONFIG["results"]['assigned'] → assigned_*.pkl (append versioned file)
        - config.CONFIG["results"]['left'] → left.pkl (overwrite)

    Expected schema for left.pkl:
        - columns: ["rid", "iti_id", ... other columns]
        - rid: int, transaction record identifier
        - iti_id: int, itinerary identifier

    --------

    :param rid_iti_pairs:
        A list of tuples, each tuple is (rid, iti_id).
        Or a 2D NumPy array with shape (N, 2), each row is [rid, iti_id].
    :type rid_iti_pairs: np.ndarray | list[tuple[int, int]]

    --------
    Example:

    >>> rid_iti_pairs = np.array([[123, 4], [456, 2], [789, 1]])
    >>> assign_feas_iti_to_trajectory(rid_iti_pairs)

    After calling:
        - assigned_*.pkl will be generated with rows with rid in [123, 456, 789].
        - left.pkl will no longer contain rid in [123, 456, 789].
    """
    fi_left = read_(config.CONFIG["results"]['left'], show_timer=False)

    assigned_df = fi_left.merge(
        pd.DataFrame(rid_iti_pairs, columns=["rid", "iti_id"]),
        on=["rid", "iti_id"],
        how="inner"
    )

    # save feas_iti_assigned
    save_(fn=config.CONFIG["results"]["assigned"],
          data=assigned_df, auto_index_on=True, verbose=False)

    # override and save feas_iti_left
    remaining_df = fi_left[~fi_left["rid"].isin(assigned_df["rid"])]
    save_(fn=config.CONFIG["results"]["left"],
          data=remaining_df, auto_index_on=False, verbose=False)
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
    save_(fn=config.CONFIG["results"]["left"],
          data=fi_left, auto_index_on=False)

    return


def _prepare_dis_penal_prob(left_df, assigned_df, dis_attached_iti_from_file, overload_train_section):
    """
    Pipeline: recompute penalties of in_vehicle links and iti probabilities.

    Recomputes probability table for current left_df, given current overload_train_section.
    """
    dis_attached_iti = filter_dis_file(
        dis_df_from_file=dis_attached_iti_from_file, left=left_df)

    penal_mapper_df = build_penal_mapper_df(
        overload_train_section=overload_train_section,
        penal_func_type=config.CONFIG["parameters"]["penalty_type"],
        penal_agg_method=config.CONFIG["parameters"]["penalty_agg_method"]
    )
    penalized_iti = cal_in_vehicle_penal_all(
        penal_mapper_df=penal_mapper_df, left=left_df)

    feas_iti_prob = compute_itinerary_probabilities(
        dis_attached_iti=dis_attached_iti,
        penalized_iti=penalized_iti
    )

    most_probable_iti = filter_most_probable_iti(
        feas_iti_prob=feas_iti_prob,
        sort_prob=True,
        exclude_non_unique_max_prob=True
    )

    return feas_iti_prob, most_probable_iti


def _select_batch(most_probable_iti, batch_size):
    """
    Select batch_size top-rid-iti pairs.
    """
    to_assign_probable_iti = most_probable_iti.head(batch_size)  # [rid, iti_id, prob]
    rid_iti_pairs = to_assign_probable_iti[['rid', 'iti_id']].values

    prob_stats = to_assign_probable_iti['prob'].quantile(
        [0, 0.25, 0.5, 0.75, 1.0]) * 100
    print(f"[INFO] Selecting batch {len(to_assign_probable_iti)}: "
          f"Prob range min={prob_stats.iloc[0]:.4f}%, Q1={prob_stats.iloc[1]:.4f}%, "
          f"median={prob_stats.iloc[2]:.4f}%, Q3={prob_stats.iloc[3]:.4f}%, max={prob_stats.iloc[4]:.4f}%")

    return rid_iti_pairs


def _preassign_get_overload(left_df, assigned_df, rid_iti_pairs):
    """
    Simulate pre-assigning rid_iti_pairs and check resulting overload.
    """
    pre_assign_df = left_df.merge(pd.DataFrame(rid_iti_pairs, columns=[
                                  "rid", "iti_id"]), on=["rid", "iti_id"], how="inner")
    assigned_plus_pre = pd.concat(
        [assigned_df, pre_assign_df], ignore_index=True)

    overload_train_section = find_overload_train_section(
        assigned=assigned_plus_pre)

    return overload_train_section


def _finalize_assignment(rid_iti_pairs, penalized_iti_in_pre_assign, feas_iti_prob, overload_batch_size):
    """
    Assign safe rid + top N overload rid.

    Notes
    -----
    - Uses feas_iti_prob (full prob table) instead of most_probable_iti (single best per rid)
      → ensures every [rid, iti_id] can get prob merged.
    """
    overload_rids_sorted = (
        penalized_iti_in_pre_assign[['rid', 'iti_id']]
        .drop_duplicates(['rid', 'iti_id'])
        .merge(feas_iti_prob.reset_index()[['rid', 'iti_id', 'prob']], on=["rid", "iti_id"], how="left")
        .sort_values(by='prob', ascending=False)
    )

    selected_overload_pairs = overload_rids_sorted.head(
        overload_batch_size)[['rid', 'iti_id']].values

    overload_rids = penalized_iti_in_pre_assign['rid'].unique()
    safe_rids = np.setdiff1d(rid_iti_pairs[:, 0], overload_rids)
    safe_mask = np.isin(rid_iti_pairs[:, 0], safe_rids)
    safe_pairs = rid_iti_pairs[safe_mask]

    final_assign_pairs = np.vstack([safe_pairs, selected_overload_pairs])

    print(
        f"[INFO] → Assigning {len(safe_pairs)} safe rids + {len(selected_overload_pairs)} overload rids = total {len(final_assign_pairs)} rids.")

    assign_feas_iti_to_trajectory(rid_iti_pairs=final_assign_pairs)


def dynamic_assignment():
    print(f"[INFO] Start dynamic assignment...")
    bs = 20_0000  # batch size
    o_bs = 100  # overload batch size

    dis_attached_iti_from_file = read_(
        config.CONFIG["results"]["dis"], show_timer=False)
    left = read_(config.CONFIG["results"]["left"], show_timer=False)
    assigned = read_all(config.CONFIG["results"]["assigned"], show_timer=False)
    overload_train_section = find_overload_train_section(assigned=assigned)

    feas_iti_prob, most_probable_iti = _prepare_dis_penal_prob(
        left, assigned, dis_attached_iti_from_file, overload_train_section)

    while not left.empty:
        rid_iti_pairs = _select_batch(most_probable_iti, bs)

        overload_train_section = _preassign_get_overload(
            left, assigned, rid_iti_pairs)

        if len(overload_train_section) == 0:
            print("[INFO] No overload → commit full batch")
            assign_feas_iti_to_trajectory(rid_iti_pairs)
        else:
            # cols: rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts, penalty
            penalized_iti_in_pre_assign = pd.merge(
                left=left[left['rid'].isin(rid_iti_pairs[:, 0])],
                right=build_penal_mapper_df(
                    overload_train_section, 
                    penal_func_type=config.CONFIG["parameters"]["penalty_type"],
                    penal_agg_method=config.CONFIG["parameters"]["penalty_agg_method"],
                ),
                on=["train_id", "board_ts", "alight_ts"],
                how="left"
            )
            # remove not-penalized rid-iti pairs
            penalized_iti_in_pre_assign.dropna(subset=["penalty"], inplace=True)
            overload_rid_count = penalized_iti_in_pre_assign["rid"].nunique()

            if overload_rid_count <= o_bs:
                print(
                    f"[INFO] overload count {overload_rid_count} ≤ {o_bs} → commit full batch")
                assign_feas_iti_to_trajectory(rid_iti_pairs)
            else:
                print(
                    f"[INFO] overload count {overload_rid_count} > {o_bs} → commit partial")
                _finalize_assignment(
                    rid_iti_pairs, penalized_iti_in_pre_assign, feas_iti_prob, o_bs)

        left = read_(config.CONFIG["results"]["left"], show_timer=False)
        assigned = read_all(
            config.CONFIG["results"]["assigned"], show_timer=False)
        most_probable_iti = most_probable_iti[most_probable_iti["rid"].isin(assigned["rid"]) == False]

        # if input("Continue? (input 'n' to break): ") == "n":
        #     break
        # only recompute prob if overload detected
        if len(overload_train_section) > 0:
            feas_iti_prob, most_probable_iti = _prepare_dis_penal_prob(
                left, assigned, dis_attached_iti_from_file, overload_train_section)


if __name__ == '__main__':
    config.load_config()
    dynamic_assignment()
    pass
