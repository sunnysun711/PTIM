"""
This script executes the full pipeline for passenger itinerary generation and categorization
based on k-shortest path and train timetable data.

Key Steps:
1. find_feas_iti_all:
    - Generates feas_iti.pkl: All feasible itineraries for passengers.
    - Outputs AFC_no_iti.pkl: Records without feasible paths.
2. _plot_check_feas_iti:
    - Visual debug utility to visualize feasible trains for a single passenger.

Usage:
- Run main() to generate itineraries.
- Optionally use _plot_check_feas_iti() to inspect specific cases.

Dependencies:
- src.passenger
- globals (K_PV_DICT, AFC)

Data Sources:
- AFC.pkl
- path.pkl
- pathvia.pkl
- TT.pkl
"""
import numpy as np

from src.passenger import find_feas_iti_all


def _plot_check_feas_iti(rid: int = None):
    """Test function for feasible itineraries found."""
    from src.passenger import AFC, K_PV_DICT, plot_seg_trains, find_feas_iti

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
          "======================================================================================\n"
          "[INFO] This script finds feasible itineraries for passengers and generates two files:\n"
          "       1. feas_iti.pkl: columns are: ['rid', 'iti_id', 'path_id','seg_id', 'train_id',\n"
          "                        'board_ts', 'alight_ts'].\n"
          "       2. AFC_no_iti.pkl: structure same as AFC. \n"
          "======================================================================================"
          "\033[0m")
    find_feas_iti_all()  # takes 10 minutes to run
    return


if __name__ == '__main__':
    # main()
    # _plot_check_feas_iti()
    pass
