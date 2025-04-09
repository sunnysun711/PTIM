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
import numpy as np

from src.passenger import find_feas_iti_all
from src.trajectory import split_feas_iti


def _plot_check_feas_iti(rid: int = None):
    """Test function for feasible itineraries found."""
    from src.globals import AFC, K_PV_DICT
    from src.passenger import plot_seg_trains, find_feas_iti

    if rid is None:
        rid, uid1, ts1, uid2, ts2 = AFC[np.random.choice(len(AFC))].flatten().tolist()
    else:
        rid, uid1, ts1, uid2, ts2 = AFC[AFC[:, 0] == rid].flatten().tolist()
    # ts1, ts2 = 20000, 23000
    print(rid, uid1, uid2, ts1, ts2)

    k_pv = K_PV_DICT[(uid1, uid2)]
    print(k_pv)

    iti_list = find_feas_iti(k_pv, ts1, ts2)
    print(len(iti_list))

    plot_seg_trains(k_pv, ts1, ts2)
    return


def main():
    print("\033[33m"
          "======================================================================================"
          "[INFO] This script finds feasible itineraries for passengers and splits them into \n"
          "       three categories:\n"
          "       1. feas_iti_assigned.pkl: Only one feasible itinerary per rid.\n"
          "       2. feas_iti_stashed.pkl: More than `feas_iti_cnt_limit` feasible itineraries \n"
          "          per rid.\n"
          "       3. feas_iti_left.pkl: Less than `feas_iti_cnt_limit` but more than 1 feasible \n"
          "          itineraries per rid.\n"
          "======================================================================================"
          "\033[0m")
    print("\033[33m"
          "[INFO] Running find_feas_iti_all and split_feas_iti(1000)...\n"
          "\033[0m")
    find_feas_iti_all(save_fn="feas_iti")  # takes 10 minutes to run
    split_feas_iti(feas_iti_cnt_limit=1000)  # split feas_iti.pkl into three files
    return


if __name__ == '__main__':
    # main()
    # _plot_check_feas_iti()
    pass
