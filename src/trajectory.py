# Make sure the "feas_iti.pkl" is generated via passenger.find_feas_iti_all(save_fn="feas_iti") before running this
# script.
# It is used in the main file to assign feasible itineraries to trajectories.
# Mostly handling itinerary files.
from src.utils import read_data, file_auto_index_saver, file_saver


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


def main():
    ...


if __name__ == '__main__':
    main()
    pass
