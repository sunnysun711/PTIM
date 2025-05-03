from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from src import config
from src.utils import read_

# this module is used to calculate the congestion penalty
# which is expressed as a dict of {(train_id, board_ts, alight_ts): penalty_value (0 to 1 float)}
# the congestion penalty is calculated by the following formula:
# penalty = \Pi_{i=1}^n (penal_func(train_load_i))


def penal_func(x: float | int | np.ndarray, capacities: tuple[float, float], type_: str = "x", k: float = None) -> float | np.ndarray:
    """
    Calculate the congestion penalty.

    :param x: The load of the train.
    :type x: float | int | np.ndarray
    :param capacities: The normal capacity and max capcity of the train.
    :type capacities: tuple[float, float]
    :param type_: The type of the congestion penalty.
    :type type_: str, optional, default="x"
    :param k: The k value of the congestion penalty.
    :type k: float, optional, default=None

    :return: The congestion penalty.
    :rtype: float | np.ndarray

    Example:
    --------
    >>> penal_func(1845, (1680.0, 1980.0), type_="x")
    0.45
    """
    cn, cc = capacities  # cap for normal, cap for commuter
    x = np.atleast_1d(x).astype(float)
    if type_ == "x":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: (cc - p) / (cc - cn), 0]
        )
    elif type_ == "x2":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: 1 - ((p - cn) / (cc - cn)) ** 2, 0]
        )
    elif type_ == "y2":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: ((p - cc) / (cn - cc)) ** 0.5, 0]
        )
    elif type_ == "normal":
        penal = np.piecewise(
            x,
            [
                x <= cn, (x > cn) & (x < cc), x >= cc
            ],
            [
                1,
                lambda p: (1 - st.norm.sf(k * (p * 2 - cc - cn) /
                           (cn - cc)) - st.norm.sf(k)) / (1 - 2 * st.norm.sf(k)),
                0
            ]
        )
    else:
        raise ValueError(f"Penalty type {type_} not supported!")
    return penal if penal.size > 1 else penal[0]


def plot_penal_funcs():
    import matplotlib
    matplotlib.use('TkAgg')
    
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4), facecolor="white")
    cn, cc = 10, 20
    x = np.linspace(cn-2, cc+2, num=1000)
    p_x = penal_func(x, (cn, cc), type_="x")
    p_x2 = penal_func(x, (cn, cc), type_="x2")
    p_y2 = penal_func(x, (cn, cc), type_="y2")
    p_normal_k1 = penal_func(x, (cn, cc), type_="normal", k=1)
    p_normal_k3 = penal_func(x, (cn, cc), type_="normal", k=3)
    p_normal_k5 = penal_func(x, (cn, cc), type_="normal", k=5)
    
    ax.plot(x, p_x, label="x")
    ax.plot(x, p_x2, label="x2")
    ax.plot(x, p_y2, label="y2")
    ax.plot(x, p_normal_k1, label="normal k1")
    ax.plot(x, p_normal_k3, label="normal k3")
    ax.plot(x, p_normal_k5, label="normal k5")
    
    ax.set_xticks([cn, cc])
    ax.set_xticklabels(["Normal", "Commuters"])
    ax.set_xlabel("Train load / Standing passenger density")
    
    ax.set_ylabel("Penalty value")
    
    ax.legend(title="Penalty function types")
    plt.show()
    return
    
    
    
    


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


if __name__ == "__main__":
    plot_penal_funcs()
    pass