# Find feasible itineraries for all AFC passenger trips based on k-shortest paths and timetable data.
# This script will generate:
#   1. feas_iti.pkl, and
#   2. optionally AFC_feas_iti_not_found.pkl (if any passengers have no feasible path).

from src.passenger import *
from src.trajectory import *

def main():
    # find_feas_iti_all(save_fn="feas_iti")  # takes 10 minutes to run
    split_feas_iti(feas_iti_cnt_limit=1000)  # split feas_iti.pkl into three files
    return

if __name__ == '__main__':
    main()
    pass
