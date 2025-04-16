from src.trajectory import split_feas_iti


def main():
    print("\033[33m"
          "======================================================================================\n"
          "[INFO] This script splits feasible itineraries into three categories:\n"
          "       1. assigned_1.pkl: Only one feasible itinerary per rid.\n"
          "       2. stashed.pkl: More than `feas_iti_cnt_limit` feasible itineraries per rid.\n"
          "       3. left.pkl: Less than `feas_iti_cnt_limit` but more than 1 feasible itineraries \n"
          "          per rid.\n"
          "======================================================================================"
          "\033[0m")
    split_feas_iti()  # split feas_iti.pkl into three files


if __name__ == '__main__':
    # main()
    pass
