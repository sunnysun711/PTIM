"""
This script executes the full pipeline for passenger itinerary generation and categorization
based on k-shortest path and train timetable data.

Key Steps:
1. Generates feas_iti.pkl: All feasible itineraries for passengers.
2. Outputs AFC_no_iti.pkl: Records without feasible paths.

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
