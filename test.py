# to test and implement GPT-generated code
import numpy as np
import pandas as pd

from src.globals import AFC, K_PV
from src.utils import read_data, file_saver


if __name__ == '__main__':
    from src.walk_time_dis import get_egress_time_from_feas_iti_left
    data = get_egress_time_from_feas_iti_left()
    print(data[np.random.choice(len(data), size=10)])
    pass
