import numpy as np
import pandas as pd

from src import config
from src.utils import save_
from src.walk_time_dis import get_egress_time_from_feas_iti_left, get_egress_time_from_feas_iti_assigned


def save_egress_times(save_on: bool = False) -> pd.DataFrame:
    """
    Save a combined DataFrame of egress times from left.pkl and assigned_1.pkl.
    "rid" as index.
    ["node1", "node2", "alight_ts", "ts2", "egress_time"] as columns.
    """
    df1 = get_egress_time_from_feas_iti_left()
    df2 = get_egress_time_from_feas_iti_assigned()
    df = pd.concat([df1, df2], ignore_index=True)
    if save_on:
        save_(fn=config.CONFIG["results"]["egress_times"], data=df, auto_index_on=True)
    return df


def plot_egress_times():
    pass


def main():
    print("\033[33m"
          "======================================================================================\n"
          "[INFO] This script generate egress time data file in `egress_times_1.pkl` as follows:\n"
          "       1. index: rid.\n"
          "       2. columns: [\"node1\", \"node2\", \"alight_ts\", \"ts2\", \"egress_time\"].\n"
          "======================================================================================"
          "\033[0m")
    save_egress_times(save_on=True)
    pass


if __name__ == '__main__':
    # main()
    pass
