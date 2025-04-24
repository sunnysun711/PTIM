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
import time

import numpy as np

from src import config


def _plot_check_feas_iti(rid: int = None):
    """Test function for feasible itineraries found."""
    from src.passenger import plot_seg_trains, find_feas_iti
    from src.globals import get_afc, get_k_pv_dict

    afc = get_afc()
    k_pv_dict = get_k_pv_dict()

    if rid is None:
        rid, uid1, ts1, uid2, ts2 = afc[np.random.choice(len(afc))].flatten().tolist()
    else:
        rid, uid1, ts1, uid2, ts2 = afc[afc[:, 0] == rid].flatten().tolist()
    # ts1, ts2 = 20000, 23000
    print(rid, uid1, uid2, ts1, ts2)

    k_pv = k_pv_dict[(uid1, uid2)]
    print(k_pv)

    iti_list = find_feas_iti(k_pv, ts1, ts2)
    print(len(iti_list))

    plot_seg_trains(k_pv, ts1, ts2)
    return


def main():
    from src.passenger import find_feas_iti_all, find_feas_iti_all_parallel
    print("\033[33m"
          "======================================================================================\n"
          "[INFO] This script finds feasible itineraries for passengers and generates two files:\n"
          "       1. feas_iti.pkl: columns are: ['rid', 'iti_id', 'path_id','seg_id', 'train_id',\n"
          "                        'board_ts', 'alight_ts'].\n"
          "       2. AFC_no_iti.pkl: structure same as AFC. \n"
          "======================================================================================"
          "\033[0m")
    df = find_feas_iti_all(save_feas_iti=False, save_afc_no_iti=False)  # 546s [08:27<00:00, 3964.37it/s] (2025-04-21)
    # df = find_feas_iti_all_parallel(save_feas_iti=False, save_afc_no_iti=False, n_jobs=-1, chunk_size=5000)
    print(df.shape)
    return


if __name__ == '__main__':
    config.load_config()
    # main()
    pass
