# Find feasible itineraries for all AFC passenger trips based on k-shortest paths and timetable data.
# This script will generate:
#   1. feas_iti.pkl, and
#   2. optionally AFC_feas_iti_not_found.pkl (if any passengers have no feasible path).

from src.passenger import *

if __name__ == '__main__':
    find_feas_iti_all(save_fn="feas_iti")
    pass
