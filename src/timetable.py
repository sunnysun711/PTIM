import os
import numpy as np
import pandas as pd

from src import config
from src.utils import read_all
from src.globals import get_tt

TI2C: dict[int, tuple[float, float]] = None


def get_ti2c() -> dict[int, tuple[float, float]]:
    """
    Get train_id to capacity mapper. Including capacities of normal passengers and commuters.

    :return: 
        train_id to capacity mapper.
        {train_id: (cap, cap_max)}
    :rtype: dict[int, tuple[float, float]]
    """
    global TI2C
    if TI2C is None:
        tt = get_tt()[:, :3]  # ["TRAIN_ID", "STATION_NID", "LINE_NID", ... ]
        typeA_cap = config.CONFIG["parameters"]["TRAIN_A_AREA"] * \
            config.CONFIG["parameters"]["STD_NORMAL"] + \
            config.CONFIG["parameters"]["TRAIN_A_SEAT"]
        typeA_cap_max = config.CONFIG["parameters"]["TRAIN_A_AREA"] * \
            config.CONFIG["parameters"]["STD_COMMUTER"] + \
            config.CONFIG["parameters"]["TRAIN_A_SEAT"]
        typeB_cap = config.CONFIG["parameters"]["TRAIN_B_AREA"] * \
            config.CONFIG["parameters"]["STD_NORMAL"] + \
            config.CONFIG["parameters"]["TRAIN_B_SEAT"]
        typeB_cap_max = config.CONFIG["parameters"]["TRAIN_B_AREA"] * \
            config.CONFIG["parameters"]["STD_COMMUTER"] + \
            config.CONFIG["parameters"]["TRAIN_B_SEAT"]

        TI2C = {}
        for train_id in np.unique(tt[:, 0]):
            line_nid = tt[tt[:, 0] == train_id][0, 2]
            if line_nid in config.CONFIG["parameters"]["A_LINES"]:
                TI2C[train_id] = (typeA_cap, typeA_cap_max)
            elif line_nid in config.CONFIG["parameters"]["B_LINES"]:
                TI2C[train_id] = (typeB_cap, typeB_cap_max)
            else:
                raise ValueError(f"line_nid {line_nid} not supported!")
    return TI2C


def reset_ti2c():
    """
    Reset TI2C to None.
    Used for testing.
    """
    global TI2C
    TI2C = None


def convert_load_to_density(data: np.ndarray, train_type: str = "a") -> np.ndarray:
    """
    Calculate SPD (Standing Passenger Density) based on in-vehicle passenger number.

    Used a trick to set less-than-0 value to 0 without processing np.NaN values in the array

        def compare_nan_array(func, a, thresh):
            out = ~np.isnan(a)
            out[out] = func(a[out] , thresh)
            return out

    :param data: passenger count on trains
    :type data: np.ndarray
    :param train_type: string, "a" or "b", defaults to "a"
    :type train_type: str, optional
    :return: SPD of the train, same shape as `data`
    :rtype: np.ndarray

    Example:
        >>> data = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        >>> train_type = "a"
        >>> convert_load_to_density(data, train_type)
    """
    if train_type not in ["a", "b"]:
        raise TypeError("train_type must be either a or b!")

    train_seats = config.CONFIG["parameters"][f"TRAIN_{train_type.upper()}_SEAT"]
    train_area = config.CONFIG["parameters"][f"TRAIN_{train_type.upper()}_AREA"]

    data = ((data - train_seats) / train_area).astype(float)
    out = ~np.isnan(data)
    out[out] = np.less(data[out], 0)
    data[out] = 0
    return data


def calculate_train_load_profile(train_id: int, board_records: np.ndarray = None) -> np.ndarray:
    """
    Calculate train load for a given train_id.

    :param train_id: 
    :type train_id: int

    :param board_records: boarded records from assigned pkl files, columns are: [board_ts, alight_ts]

        Default is None, which will read and process from all assigned_*.pkl files.
    :type board_records: np.ndarray, optional

    :return:
        columns are: ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']

        where 'load' is the number of passengers on the train in the section.
    :rtype: np.ndarray
    """
    if board_records is None:
        assigned = read_all(
            config.CONFIG["results"]["assigned"], show_timer=False)
        board_records = assigned[assigned["train_id"] ==
                                 train_id][["board_ts", "alight_ts"]].values

    board_ts, counts = np.unique(board_records[:, 0], return_counts=True)
    board_dic: dict[int, int] = dict(
        zip(board_ts, counts))  # board_dic.get(key, 0)
    alight_ts, counts = np.unique(board_records[:, 1], return_counts=True)
    alight_dic: dict[int, int] = dict(zip(alight_ts, counts))

    tt = get_tt()
    tt = tt[tt[:, 0] == train_id][:, [1, 4, 5]]

    this_tt = np.hstack([tt[:-1, [0, 2]], tt[1:, [0, 1]]]
                        )  # sta1, dep_ts, sta2, arr_ts
    on_train_psg_num = 0
    sec_psg = []
    for sta1, arr_ts, dep_ts in tt:
        alight_num = alight_dic.get(arr_ts, 0)
        board_num = board_dic.get(dep_ts, 0)
        on_train_psg_num += board_num - alight_num
        sec_psg.append(on_train_psg_num)
    if sec_psg[-1] != 0:
        print(
            f"\033[31m[ERROR] Train {train_id} is not empty at the end of the trip!\033[0m")
        raise ValueError(
            "Num of passengers boarding and alighting should be equal!")
    sec_psg = sec_psg[:-1]
    this_tt = np.hstack([this_tt, np.array(sec_psg).reshape(-1, 1)])
    return this_tt


def find_overload_train_section(assigned: pd.DataFrame = None, suppress_warning: bool = False) -> dict[int, np.ndarray]:
    """
    Find the overload train, section pairs.

    :param assigned: assigned trajectory dataframe. 
        Defaults to None, meaning use `read_all()` to concat all assigned_*.pkl files.

        Expected columns: [`rid`, `iti_id`, `path_id`, `seg_id`, `train_id`, `board_ts`, `alight_ts`]
    :type assigned: pd.DataFrame, optional, default=None
    
    :param suppress_warning: Whether to suppress warning messages for trains exceeding max capacity.
        If True, no warning will be printed. If False, a warning will be printed for each train exceeding max capacity.
    :type suppress_warning: bool, optional, default=False

    :return: Dictionary mapping train_id to np.ndarray, columns are: 

        [`sta1_nid`, `dep_ts`, `sta2_nid`, `arr_ts`, `load`]

        where `load` is the number of passengers on the train in the section.

        Only overloaded train_id will be included in the dictionary.

        Only overloaded sections will be included in the np.ndarray.
    """
    if assigned is None:
        assigned = read_all(
            config.CONFIG["results"]["assigned"], show_timer=False)

    res: dict[int, np.ndarray] = {}
    for train_id, board_counts in assigned["train_id"].value_counts().items():
        cap, cap_max = get_ti2c()[train_id]

        if board_counts < cap:  # all boarded less than normal capacity, skip
            continue

        # ['sta1_nid', 'dep_ts','sta2_nid', 'arr_ts', 'load']
        board_records_this_train = assigned[assigned["train_id"] == train_id][[
            "board_ts", "alight_ts"]].values
        load_arr = calculate_train_load_profile(
            train_id, board_records=board_records_this_train)
        if np.all(load_arr[:, -1] <= cap):  # section passengers less than capacity, skip
            continue

        if np.any(load_arr[:, -1] > cap_max) and not suppress_warning:
            print(
                f"\033[33m[WARNING] Train {train_id} exceeds max capacity! {load_arr[:, -1].max()} > {cap_max}\033[0m")

        res[train_id] = load_arr[load_arr[:, -1] > cap]

    return res


def plot_timetable(li: int = 2, upd: list = None, show_load=True, save_subfolder="", assigned: pd.DataFrame = None):
    """
    Plot train timetable of specified line and updowns. Optionally show train 
    load based on current assigned pkl files.

    :param li: line number, defaults to 2
    :type li: int, optional, default=2

    :param upd: list of updowns, should be either [-1] or [1] or [-1, 1].
    :type upd: list, optional, default=[-1, 1]

    :param show_load: whether to show train load, defaults to True
    :type show_load: bool, optional, default=True

    :param save_subfolder: subfolder to save the plot, defaults to "", meaning do not save.
    :type save_subfolder: str, optional, default=""

    :param assigned: dataframe of assigned trajectories. 
        Expected columns: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts]
    :type assigned: pd.DataFrame

    """
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from src.utils import ts2tstr, tstr2ts

    # initialize
    upd = [1, -1] if upd is None else upd
    prd = ["05:45", "23:50"]
    prd_t = [tstr2ts(prd[0]), tstr2ts(prd[1])]

    tt = pd.DataFrame(
        get_tt(), columns=["TRAIN_ID", "STATION_NID", "LINE_NID", "UPDOWN", "ARRIVE_TS", "DEPARTURE_TS"]
    ).set_index("TRAIN_ID")
    tt = tt[
        (tt["LINE_NID"] == li) &
        (tt["UPDOWN"].isin(upd)) &
        (tt["ARRIVE_TS"].isin(range(prd_t[0]-600, prd_t[1]+600)))
    ]

    # background of timetable
    fig, ax = plt.subplots(1, 1, facecolor="white", figsize=(24, 6), dpi=200)

    # set x ticks and labels
    ax.set_xlabel('Time', fontsize=8)
    ax.set_xlim(prd_t[0], prd_t[1])
    xticks = list(range(prd_t[0] // 3600 * 3600,
                  3600 * (prd_t[1] // 3600) + 3600, 600))
    xticklabels = [ts2tstr(i) for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    # set vertical lines for time
    for i in range(prd_t[0] // 600 * 600 - 1200, prd_t[1] + 1200, 600):
        if i % 3600 == 0:  # Hour
            ax.axvline(i, ls="-", lw=0.8, c="limegreen")
        elif i % 1800 == 0:  # Half-hour
            ax.axvline(i, ls="-", lw=0.4, c="limegreen")
        else:  # 10-minute
            ax.axvline(i, ls="--", lw=0.2, c="limegreen")

    # set y ticks and labels
    ax.set_ylabel("Stations", fontsize=8)
    if li != 7:
        stations = np.sort(tt["STATION_NID"].unique())
        stations_labels = stations
    else:
        stations = list(
            np.sort(tt["STATION_NID"].unique())
        ) + [tt["STATION_NID"].max()+1]
        stations_labels = stations.copy()
        stations_labels[-1] = stations_labels[0]
    # set horizontal lines for stations
    for station_nid in stations:
        ax.axhline(station_nid, ls=":", lw=0.4, c="limegreen")
    ax.set_yticks(stations, labels=stations_labels, fontsize=8)
    ax.set_ylim(min(stations)-0.5, max(stations)+0.5)

    # plot train lines
    segments: list[list[tuple[int, int], tuple[int, int]]] = []
    load_values = []

    for tid in tt.index.unique():
        train_data = tt[tt.index == tid]
        if train_data.shape[0] < 3 or tid == 10200746:  # Line 2 exception.
            continue

        if show_load:
            board_records = assigned[assigned["train_id"]
                                     == tid][["board_ts", "alight_ts"]].values
            load_data = calculate_train_load_profile(
                train_id=tid, board_records=board_records)
        else:
            # 0 if not showing load
            load_data = np.zeros((train_data.shape[0]-1, 2))

        for i in range(len(train_data) - 1):
            # Connect the current station's departure with the next station's arrival
            row1 = train_data.iloc[i]
            row2 = train_data.iloc[i + 1]

            dep_ts, arr_ts = row1['DEPARTURE_TS'], row2['ARRIVE_TS']
            sta1, sta2 = row1['STATION_NID'], row2['STATION_NID']

            # Special case for line 7: Handle circular line where the first and last stations are connected.
            if li == 7 and min(sta1, sta2) == min(stations) and max(sta1, sta2) == max(stations) - 1:
                if sta1 < sta2:
                    sta1 = max(stations)
                else:
                    sta2 = max(stations)

            load_values.append(load_data[i, -1])

            # Store the line segments for plotting
            segments.append(
                [
                    (dep_ts, sta1),
                    (arr_ts, sta2)
                ]
            )

    cap, cap_max = get_ti2c()[tid]
    # create linecollection
    lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, cap_max*1.3))  # to see severe overload situations
    # Set the values used for colormapping
    lc.set_array(load_values)
    lc.set_linewidth(0.8)
    line = ax.add_collection(lc)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cb = fig.colorbar(line, ax=ax, extend="max", cax=cax)
    cb.set_label("Pax on train")
    cb.ax.tick_params(axis='y', direction='in', length=2, labelsize=6)
    plt.tight_layout(pad=1)

    if save_subfolder != "":
        saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
        if save_subfolder and not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        upd_str = "_".join([str(i) for i in upd])
        load_str = "on" if show_load else "off"
        fig.savefig(f"{saving_dir}/TT_{li:02}_UPD_{upd_str}_LOAD_{load_str}.pdf",
                    bbox_inches="tight", dpi=600)
        fig.savefig(f"{saving_dir}/TT_{li:02}_UPD_{upd_str}_LOAD_{load_str}.png",
                    bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_timetable_all(save_subfolder: str, separate_upd: bool = False, assigned: pd.DataFrame = None):

    saving_dir = config.CONFIG["figure_folder"] + "/" + save_subfolder
    if save_subfolder and not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    print(f"[INFO] Plotting timetables and saving figures to {saving_dir}...")

    assigned = read_all(config.CONFIG["results"]["assigned"],
                        show_timer=False) if assigned is None else assigned

    for line in [1, 2, 3, 4, 7, 10]:
        print(f"[INFO] Plotting timetable for Line {line}...")
        if separate_upd:
            plot_timetable(
                li=line, upd=[-1], show_load=True, save_subfolder=save_subfolder, assigned=assigned)
            plot_timetable(
                li=line, upd=[1], show_load=True, save_subfolder=save_subfolder, assigned=assigned)
        else:
            plot_timetable(
                li=line, upd=[-1, 1], show_load=True, save_subfolder=save_subfolder, assigned=assigned)


if __name__ == '__main__':
    for version in ["_greedy", "1", "2"]:
        config.load_config(f"configs/config{version}.yaml")
        print("\n === Version: ", version, "===\n")
        
        plot_timetable_all(f"TT_config{version}", separate_upd=True)

        overload_info = find_overload_train_section(suppress_warning=False)  # will print warnings
        # print(overload_info)
    
