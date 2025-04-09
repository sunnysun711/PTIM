"""
This script executes the full pipeline for passenger itinerary generation and categorization
based on k-shortest path and train timetable data.

Key Steps:
1. find_feas_iti_all:
    - Generates feas_iti.pkl: All feasible itineraries for passengers.
    - Optionally outputs AFC_feas_iti_not_found.pkl: Records without feasible paths.
2. split_feas_iti:
    - Splits feas_iti.pkl into:
        - feas_iti_assigned.pkl (1 itinerary per rid)
        - feas_iti_stashed.pkl (too many itineraries, > threshold)
        - feas_iti_left.pkl (between 2 and threshold)
3. _plot_check_feas_iti:
    - Visual debug utility to visualize feasible trains for a single passenger.

Usage:
- Run main() to generate and split itineraries.
- Optionally use _plot_check_feas_iti() to inspect specific cases.

Dependencies:
- src.passenger
- src.trajectory
- globals (K_PV_DICT, AFC)

Data Sources:
- AFC.pkl
- path.pkl
- pathvia.pkl
- TT.pkl
"""

from src.passenger import find_feas_iti_all
from src.trajectory import split_feas_iti


def _plot_check_feas_iti():
    """Test function for feasible itineraries found."""
    from globals import AFC, K_PV_DICT
    from src.passenger import plot_seg_trains, find_feas_iti

    # rid, uid1, ts1, uid2, ts2 = AFC[np.random.choice(len(AFC))].flatten().tolist()
    # rid = 505630  # 1349
    rid = 317
    rid, uid1, ts1, uid2, ts2 = AFC[AFC[:, 0] == rid].flatten().tolist()
    # ts1, ts2 = 20000, 23000
    print(rid, uid1, uid2, ts1, ts2)

    k_pv = K_PV_DICT[(uid1, uid2)]
    print(k_pv)

    iti_list = find_feas_iti(k_pv, ts1, ts2)
    print(len(iti_list))

    plot_seg_trains(k_pv, ts1, ts2)
    return


def main(feas_iti_cnt_limit=1000):
    find_feas_iti_all(save_fn="feas_iti")  # takes 10 minutes to run
    split_feas_iti(feas_iti_cnt_limit=feas_iti_cnt_limit)  # split feas_iti.pkl into three files
    return


if __name__ == '__main__':
    main()
    pass
