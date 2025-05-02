from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.utils import read_

# this module is used to calculate the congestion penalty
# which is expressed as a dict of {(train_id, board_ts, alight_ts): penalty_value (0 to 1 float)}
# the congestion penalty is calculated by the following formula:
# penalty = \Pi_{i=1}^n (penal_func(train_load_i))


def penal_func(x: float | np.ndarray, cap: float, cap_max: float, type_: str = "x", k: float = None) -> float | np.ndarray:
    """
    Calculate the congestion penalty.

    Args:
        x: The load of the train.
        cap: The capacity of the train.
        cap_max: The maximum capacity of the train.
        type_: The type of the congestion penalty.
        k: The k value of the congestion penalty.

    Returns:
        The congestion penalty.
    """
    x = np.atleast_1d(x)
    penal: np.ndarray = ...
    return penal if penal.size > 1 else penal[0]


def build_penalty_dict(overload_train_section: dict[int, np.ndarray]) -> dict[tuple[int, int, int], float]:
    """
    Build a dict of {(train_id, board_ts, alight_ts): penalty_value (0 to 1 float)}
    from the overload train section dict.

    Args:
        overload_train_section: The overload train section dict. {train_id: np.ndarray} \n
            columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load'] (only overload sections) \n
            should get this variable from src.timetable.find_overload_train_section()

    Returns:
        The penalty dict.
    """
    penalty_dict: dict[tuple[int, int, int], float] = {}

    ...
    return penalty_dict



