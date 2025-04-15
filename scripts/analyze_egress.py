import numpy as np
import pandas as pd

from src import config
from src.utils import save_
from src.walk_time_dis import get_egress_time_from_feas_iti_left, get_egress_time_from_feas_iti_assigned


def save_egress_times(save_on: bool = False) -> pd.DataFrame:
    """
    Save a combined DataFrame of egress times from feas_iti_left and feas_iti_assigned.
    "rid" as index.
    ["node1", "node2", "alight_ts", "ts2", "egress_time"] as columns.
    """
    df1 = get_egress_time_from_feas_iti_left()
    df2 = get_egress_time_from_feas_iti_assigned()
    df = pd.concat([df1, df2], ignore_index=True)
    if save_on:
        save_(fn=config.CONFIG["results"]["egress_times"], data=df, auto_index_on=True)
    return df


def main():
    # save_egress_times(save_on=True)
    pass
