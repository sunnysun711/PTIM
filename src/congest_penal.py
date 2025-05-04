from typing import Callable
import numpy as np
import pandas as pd
import scipy.stats as st

# this module is used to calculate the congestion penalty
# which is expressed as a dict of {(train_id, board_ts, alight_ts): penalty_value (0 to 1 float)}
# the congestion penalty is calculated by the following formula:
# penalty = \Pi_{i=1}^n (penal_func(train_load_i))


def penal_func(x: float | int | np.ndarray, capacities: tuple[float, float], func_type: str | float | int = "x") -> np.ndarray:
    """
    Calculate the congestion penalty.

    :param x: The load of the train.
    :type x: float | int | np.ndarray
    :param capacities: The normal capacity and max capcity of the train.
    :type capacities: tuple[float, float]
    :param func_type: The type of the congestion penalty. 
        Should be one of ["x", "x2", "x2_", "y2", "y2_"].

        If a number (float or integer) is provided, normally distributed CDF
        penalty will be used.
    :type func_type: str | float | int, optional, default="x"

    :return: The congestion penalty.
    :rtype: np.ndarray

    Example:
    --------
    >>> penal_func(1845, (1680.0, 1980.0), func_type="x")
    array([0.45])
    >>> penal_func(1845, (1680.0, 1980.0), func_type=2.5)
    array([0.40005239])
    >>> penal_func(np.random.rand(5)*(1980-1680)+1680, (1680.0, 1980.0), func_type="x2_")
    array([0.92070332, 0.05099632, 0.4538884 , 0.04286799, 0.71265908])
    """
    cn, cc = capacities  # cap for normal, cap for commuter
    x = np.atleast_1d(x).astype(float)
    if func_type == "x":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: (cc - p) / (cc - cn), 0]
        )
    elif func_type == "x2":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: 1 - ((p - cn) / (cc - cn)) ** 2, 0]
        )
    elif func_type == "x2_":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: ((p - cc) / (cc - cn)) ** 2, 0]
        )
    elif func_type == "y2":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: ((p - cc) / (cn - cc)) ** 0.5, 0]
        )
    elif func_type == "y2_":
        penal = np.piecewise(
            x,
            [x <= cn, (x > cn) & (x < cc), x >= cc],
            [1, lambda p: 1 - ((p - cn) / (cc - cn)) ** 0.5, 0]
        )
    elif isinstance(func_type, (float, int)):
        k = float(func_type)
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
        raise ValueError(f"Penalty type {func_type} not supported!")
    return penal


def plot_penal_funcs():
    import matplotlib
    matplotlib.use('TkAgg')

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white")
    cn, cc = 10, 20
    x = np.linspace(cn-2, cc+2, num=1000)
    p_x = penal_func(x, (cn, cc), func_type="x")
    p_x2 = penal_func(x, (cn, cc), func_type="x2")
    p_x2_ = penal_func(x, (cn, cc), func_type="x2_")
    p_y2 = penal_func(x, (cn, cc), func_type="y2")
    p_y2_ = penal_func(x, (cn, cc), func_type="y2_")
    # p_normal_k1 = penal_func(x, (cn, cc), func_type=1)
    p_norm1 = penal_func(x, (cn, cc), func_type=3)
    p_norm2 = penal_func(x, (cn, cc), func_type=2)

    ax.plot(x, p_x, label="x")
    ax.plot(x, p_x2, label="x2")
    ax.plot(x, p_x2_, label="x2_")
    ax.plot(x, p_y2, label="y2")
    ax.plot(x, p_y2_, label="y2_")
    # ax.plot(x, p_normal_k1, label="normal k1")
    ax.plot(x, p_norm1, label="normal k3")
    ax.plot(x, p_norm2, label="normal k2")

    ax.set_xticks([cn, cc])
    ax.set_xticklabels(["Normal", "Commuters"])
    ax.set_xlabel("Train load / Standing passenger density")

    ax.set_ylabel("Penalty value")

    ax.legend(title="Penalty function types", ncols=2)
    plt.show()
    return


def _build_penalty_dict_array(
    overload_train_section: dict[int, np.ndarray],
    penal_func_type: str | float | int = "x",
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Internal function to build a dict of detailed penalty arrays
    from the overload train section dict.

    :param overload_train_section: The overload train section dict.
        {train_id: np.ndarray}

        Columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
        (only overload sections)

        Should be generated from `src.timetable.find_overload_train_section()`
    :type overload_train_section: dict[int, np.ndarray]

    :param penal_func_type: The function type for the penalty. Should be
        one of ["x", "x2", "x2_", "y2", "y2_"] or a number for normal CDF
        penalty parameter.
    :type penal_func_type: str | float | int, optional, default="x"

    :return: The penalty dict, mapping (train_id, board_ts, alight_ts)
        to a numpy array of shape (num_passed_sections, 6),
        columns: ['sta1_nid', 'dep_ts', 'sta2_nid', 'arr_ts', 'load', 'penalty']
    :rtype: dict[tuple[int, int, int], np.ndarray]
    """
    from src.globals import get_tt
    from src.timetable import get_ti2c

    tt = get_tt()
    penalty_dict: dict[tuple[int, int, int], np.ndarray] = {}

    for train_id, overload_sections in overload_train_section.items():
        assert overload_sections.shape[0] > 0, (
            f"\033[31m[ERROR] Empty overload_sections for train_id={train_id}\033[0m",
            f"Shape: {overload_sections.shape}"
        )

        capacities = get_ti2c()[train_id]

        # Compute penalties for each overloaded section
        penalties = penal_func(
            overload_sections[:, -1],
            capacities=capacities,
            func_type=penal_func_type
        )

        # Filter timetable for this train
        tt_filtered = tt[(tt[:, 0] == train_id)]
        board_ts_arr = tt_filtered[:, -1]
        alight_ts_arr = tt_filtered[:, -2]
        n = len(tt_filtered)

        overload_board_ts_latest = overload_sections[:, 1].max()
        overload_alight_ts_earliest = overload_sections[:, 3].min()

        for i in range(n):
            board_ts = board_ts_arr[i]
            if board_ts > overload_board_ts_latest:
                break

            for j in range(i + 1, n):
                if tt_filtered[j, 1] == tt_filtered[i, 1]:
                    break

                alight_ts = alight_ts_arr[j]
                if alight_ts < overload_alight_ts_earliest:
                    break

                dep_ts_arr = overload_sections[:, 1]
                arr_ts_arr = overload_sections[:, 3]

                mask = (alight_ts > dep_ts_arr) & (board_ts < arr_ts_arr)
                if np.any(mask):
                    sections_passed = overload_sections[mask]
                    penalties_passed = penalties[mask].reshape(-1, 1)
                    combined = np.hstack([sections_passed, penalties_passed])
                    key = (train_id, board_ts, alight_ts)
                    penalty_dict[key] = combined

    return penalty_dict


def build_penalty_df(
    overload_train_section: dict[int, np.ndarray],
    penal_func_type: str | float | int = "x",
    penal_agg_method: str = "min"
) -> pd.DataFrame:
    """
    Build a DataFrame of penalty values from the overload train section dict.

    :param overload_train_section: The overload train section dict.
        {train_id: np.ndarray}

        Columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
        (only overload sections)

        Should be generated from `src.timetable.find_overload_train_section()`
    :type overload_train_section: dict[int, np.ndarray]

    :param penal_func_type: The function type for the penalty. Should be
        one of ["x", "x2", "x2_", "y2", "y2_"] or a number for normal CDF
        penalty parameter.
    :type penal_func_type: str | float | int, optional, default="x"

    :param penal_agg_method: The aggregation method for the penalty values
        across multiple overload sections. Should be one of ["min",
        "mean", "prod"].
    :type penal_agg_method: str, optional, default="min"

    :return: A DataFrame with columns ['train_id', 'board_ts', 'alight_ts', 
        'penalty'], where each row maps a (train_id, board_ts, alight_ts) 
        to its penalty value (0 to 1) aggregated using `penal_agg_method`.
    :rtype: pd.DataFrame
    """
    agg_method_mapping: dict[str, Callable] = {
        "min": np.min,
        "mean": np.mean,
        "prod": np.prod
    }
    if penal_agg_method not in agg_method_mapping:
        raise ValueError(f"Unsupported penal_agg_method: {penal_agg_method}")

    penalty_dict_array = _build_penalty_dict_array(
        overload_train_section, penal_func_type
    )

    agg_method = agg_method_mapping[penal_agg_method]
    penalty_df = pd.DataFrame(
        [
            (*key, float(agg_method(arr[:, -1])))
            for key, arr in penalty_dict_array.items()
        ],
        columns=["train_id", "board_ts", "alight_ts", "penalty"]
    )

    return penalty_df


if __name__ == "__main__":
    # plot_penal_funcs()
    # res = penal_func(606, capacities=(600, 810), func_type="x")
    from src import config
    config.load_config()

    from src.timetable import get_ti2c, reset_ti2c

    reset_ti2c()
    config.CONFIG["parameters"]["TRAIN_A_AREA"] = 50
    config.CONFIG["parameters"]["TRAIN_B_AREA"] = 50
    print(np.unique([i for i in get_ti2c().values()]))

    res = build_penalty_df(
        overload_train_section={10200956: np.array(
            [[10241, 29678, 10240, 29761, 605],
             [10240, 29796, 10239, 29862, 622],
             [10239, 29899, 10238, 30006, 620]]
        )
        },
        penal_func_type="x",
        penal_agg_method="min",
        debug=True
    )

    np.set_printoptions(precision=6, suppress=True, linewidth=180)

    for k, v in res.items():
        print(k, "\n", v)
    pass
