import numpy as np
import pandas as pd

from src.utils import file_auto_index_saver
from src.walk_time_dis import get_egress_time_from_feas_iti_left, get_egress_time_from_feas_iti_assigned


def save_egress_times():
    data1 = get_egress_time_from_feas_iti_left()
    data2 = get_egress_time_from_feas_iti_assigned()
    data = np.vstack((data1, data2))

    df = pd.DataFrame(data, columns=["node1", "node2", "alight_ts", "ts2", "egress_time"])
    file_auto_index_saver(lambda save_fn: df)(save_fn="egress_times")
    return


def main():
    pass


if __name__ == '__main__':
    main()
    pass
