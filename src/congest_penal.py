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

    :param x: The load of the train.
    :type x: float | np.ndarray
    :param cap: The capacity of the train.
    :type cap: float
    :param cap_max: The maximum capacity of the train.
    :type cap_max: float
    :param type_: The type of the congestion penalty.
    :type type_: str, optional, default="x"
    :param k: The k value of the congestion penalty.
    :type k: float, optional, default=None

    :return: The congestion penalty.
    :rtype: float | np.ndarray

    Example:
    --------
    >>> penal_func(1845, 1680.0, 1980.0, type_="x")
    0.5
    """
    x = np.atleast_1d(x)
    penal: np.ndarray = ...
    return penal if penal.size > 1 else penal[0]


def build_penalty_dict(overload_train_section: dict[int, np.ndarray]) -> dict[tuple[int, int, int], float]:
    """
    Build a dict of {(train_id, board_ts, alight_ts): penalty_value (0 to 1 float)}
    from the overload train section dict.

    :param overload_train_section: The overload train section dict. {train_id: np.ndarray} 
        
        columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load'] (only overload sections) 
        
        should generated from `src.timetable.find_overload_train_section()`
    :type overload_train_section: dict[int, np.ndarray]

    :return: The penalty dict.
    :rtype: dict[tuple[int, int, int], float]
    """
    penalty_dict: dict[tuple[int, int, int], float] = {}

    ...
    return penalty_dict



