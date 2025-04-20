"""
This module handles passenger itinerary analysis for public transit systems.
It provides functionality to find feasible train itineraries between stations within given time windows.

Key Functions:
1. find_trains: Finds available trains between two stations within a time window
2. find_seg_trains: Finds trains for each segment of a path with pruning
3. find_feas_iti: Finds all feasible itineraries for a given path
4. plot_seg_trains: Visualizes available trains for each path segment
5. find_feas_iti_all: Main function to process all passenger records

Dependencies:
- numpy: For numerical operations
- pandas: For data manipulation
- tqdm: For progress tracking

Data sources:
- path.pkl
- pathvia.pkl
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as pmap
from functools import partial

from src import config
from src.utils import ts2tstr, save_


def find_trains(nid1: int, nid2: int, ts1: int, ts2: int, line: int, upd: int) -> list[tuple[int, int, int]]:
    """
    Find trains, board_ts, alight_ts for nid pairs within ts range.
    Board_ts is obtained from departure_ts, alight_ts from arrive_ts.

    :return: [(train_id, board_ts, alight_ts), ...]
    """
    assert ts1 < ts2, f"ts1 should be smaller than ts2: {ts1}, {ts2}"

    # filter line
    from src.globals import get_tt
    tt_ = get_tt()
    start_idx, end_idx = np.searchsorted(tt_[:, 2], [line, line + 1])
    tt: np.ndarray[int] = tt_[start_idx:end_idx]

    # filter updown
    start_idx, end_idx = np.searchsorted(tt[:, 3], [upd, upd + 1])
    tt = tt[start_idx:end_idx]

    # filter od and ts range
    tt = tt[
        ((tt[:, 1] == nid1) | (tt[:, 1] == nid2))  # filter nid
        & (tt[:, 5] > ts1)  # filter ts1
        & (tt[:, 4] < ts2)  # filter ts2
        ]

    # Pair finding using vectorized comparison
    train_ids = tt[:-1, 0] == tt[1:, 0]  # check if train_id is the same for consecutive rows
    nid1_condition = tt[:-1, 1] == nid1  # check if the first station is _nid1
    nid2_condition = tt[1:, 1] == nid2  # check if the second station is _nid2

    # Combine conditions
    condition = train_ids & nid1_condition & nid2_condition

    # Prepare result using filtered indices
    trains = [(tt[i, 0], tt[i, 5], tt[i + 1, 4]) for i in range(len(tt) - 1) if condition[i]]
    return trains


def find_seg_trains(pv: np.ndarray, ts1: int, ts2: int) -> list[list[tuple[int, int, int]]]:
    """
    Find trains for each segment in pv. If for any single one segment, no trains can be found, return [].

    This function will do a backward pruning step to remove trains that cannot be used in the next segment.

    This holds: len(seg_trains) = number of seg in this path_id.

    :param pv: Path via array, [path_id, nid1, nid2, line, upd]
    :param ts1: Timestamp 1.
    :param ts2: Timestamp 2.
    :return: List of trains for each segment, each element is a list of trains,
        [[(train_id, b_ts, a_ts), (train_id, b_ts, a_ts),...], [(train_id, b_ts, a_ts)], ...].
    """
    seg_trains: list[list[tuple[int, int, int]]] = []
    ts1_this_seg = ts1
    for seg in pv:
        trains = find_trains(nid1=seg[1], nid2=seg[2], ts1=ts1_this_seg, ts2=ts2, line=seg[3], upd=seg[4])
        if not trains:  # cannot find trains in this segment, meaning no feasible itineraries of this path_id
            seg_trains = []  # return empty list
            break
        seg_trains.append(trains)
        ts1_this_seg = min([_[2] for _ in trains])  # update ts1 for next seg
    else:  # this path_id does have at least one feasible iti
        # do a backward pruning step
        ts2_this_seg = max([_[1] for _ in seg_trains[-1]])
        for trains in reversed(seg_trains[:-1]):
            trains[:] = [t for t in trains if t[2] < ts2_this_seg]  # prune
            ts2_this_seg = max([_[1] for _ in trains])
    return seg_trains


def find_feas_iti(k_pv: np.ndarray, ts1: int, ts2: int) -> list[list[int | tuple[int, int, int]]]:
    """
    Find feasible itineraries with k shortest path via array (`k_pv`) and ts range (`ts1`,`ts2`) provided.
    :param k_pv: path via array, [path_id, nid1, nid2, line, upd]
    :param ts1: timestamp 1.
    :param ts2: timestamp 2.
    :return: list of itineraries,
        [[path_id, (train_id, b_ts, a_ts), (train_id, b_ts, a_ts), ...], [path_id, (train_id, b_ts, a_ts)], ...]
    """
    feas_iti_list: list[list[int | tuple[int, int, int]]] = []
    for path_id in set(k_pv[:, 0]):
        pv = k_pv[k_pv[:, 0] == path_id]
        seg_trains: list[list[tuple[int, int, int]]] = find_seg_trains(pv=pv, ts1=ts1, ts2=ts2)
        if not seg_trains:  # no feasible iti in this path_id
            continue

        def backtrack(seg_idx: int, current_path: list[tuple[int, int, int]]):
            """Build a complete feasible itinerary from seg_trains"""
            if seg_idx == len(seg_trains):
                feas_iti_list.append([path_id] + current_path)
                return

            for train in seg_trains[seg_idx]:
                if not current_path or current_path[-1][2] < train[1]:  # check transfer time > 0
                    current_path.append(train)
                    backtrack(seg_idx + 1, current_path)  # recursion
                    current_path.pop()

        backtrack(0, [])
    return feas_iti_list


def plot_seg_trains(k_pv: np.ndarray, ts1: int, ts2: int):
    """
        Visualizes available trains for each segment of given paths.

    Args:
        k_pv: Array of path segments with shape [n,5]
        ts1: Minimum timestamp for visualization
        ts2: Maximum timestamp for visualization

    Displays:
        Interactive plot showing train connections between stations
        - Each subplot represents a different path
        - Lines connect boarding/alighting times
        - Colors represent different train lines

    Note:
        Requires matplotlib and TkAgg backend
    :param k_pv: k path via numpy array
    :param ts1: timestamp 1
    :param ts2: timestamp 2
    :return:
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    path_id_seg_trains: dict[int, list[list[tuple[int, int, int]]]] = {}
    for path_id in set(k_pv[:, 0]):
        pv = k_pv[k_pv[:, 0] == path_id]
        seg_trains: list[list[tuple[int, int, int]]] = find_seg_trains(pv=pv, ts1=ts1, ts2=ts2)
        if not seg_trains:  # this path_id does have at least one feasible iti
            continue
        path_id_seg_trains[path_id] = seg_trains

    line_colors = {
        1: '#3C2D7B',
        2: '#E45E43',
        3: '#D11E5F',
        4: '#2CAF63',
        7: '#96CDE1',
        10: '#1010A4',
    }

    # Determine the number of rows and columns based on the number of items
    num_plots = len(path_id_seg_trains)
    ncols = 2 if num_plots > 4 else 1
    nrows = (num_plots + 1) // ncols if num_plots > 4 else num_plots  # Ensure enough rows for all subplots

    # Plot with dynamic subplots
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 2.5 * nrows),
        sharex=True
    )

    if num_plots == 1:
        axs = [axs]  # Make sure axs is iterable even if there's only one subplot

    # Calculate xticks (every 10 minutes between ts1 and ts2)
    xticks = list(range(ts1 // 600 * 600 + 600, ts2 // 600 * 600 + 600, 600))  # 600 seconds = 10 minutes
    xticklabels = [ts2tstr(t, include_seconds=False) for t in xticks]

    # Add ts1 and ts2 to the xticks list and labels
    xticks = [ts1] + xticks + [ts2]
    xticklabels = [ts2tstr(ts1, include_seconds=True)] + xticklabels + [ts2tstr(ts2, include_seconds=True)]

    if ncols == 1:
        axs[-1].set_xlabel("Time")
        axs[-1].set_xlim([ts1, ts2])
        axs[-1].set_xticks(xticks)
        axs[-1].set_xticklabels(xticklabels)
        # axs[-1].tick_params(axis='x', labelrotation=45)
        for label in axs[-1].get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_verticalalignment('top')
    else:
        for i in range(ncols):
            axs[-1, i].set_xlabel("Time")
            axs[-1, i].set_xlim([ts1, ts2])
            axs[-1, i].set_xticks(xticks)
            axs[-1, i].set_xticklabels(xticklabels)
            # axs[-1, i].tick_params(axis='x', labelrotation=45)
            for label in axs[-1, i].get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
                label.set_verticalalignment('top')

    # Flatten axs for easier indexing if multiple columns
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()

    # Loop through each path_id and plot the corresponding subgraph
    for i, (path_id, seg_trains) in enumerate(path_id_seg_trains.items()):
        pv = k_pv[k_pv[:, 0] == path_id]
        ax = axs[i]
        ax.set_title(f"Path ID: {path_id}")
        ax.set_ylabel("Stations")

        # # Set the y-ticks and labels
        # ax.set_yticks(range(len(seg_trains) + 1))
        ax.set_ylim([0, len(seg_trains)])
        ax.hlines(
            y=np.arange(1, len(seg_trains)),
            xmin=ts1, xmax=ts2,
            lw=1, color='gray', alpha=0.5
        )
        # Set the y-ticks and labels
        yticks = []
        ytick_labels = []
        for i in range(len(pv)):
            yticks.append(i + 0.1)
            yticks.append(i + 0.5)
            yticks.append(i + 0.9)
            ytick_labels.append(f"{pv[i, 1]}")
            ytick_labels.append(f"({pv[i, 3]}, {pv[i, 4]})")
            ytick_labels.append(f"{pv[i, 2]}")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(axis='y', direction='in', length=0, labelsize=8)
        ax.tick_params(axis='x', direction='out', length=3, labelsize=8)

        # Plot lines to indicate train connections
        for seg_id, trains in enumerate(seg_trains):
            line = pv[seg_id, 3]
            for train in trains:
                ax.plot([train[1], train[2]], [seg_id, seg_id + 1], lw=1, color=line_colors.get(line, 'black'))

    plt.tight_layout()
    plt.show()
    return


def find_feas_iti_all(save_feas_iti: bool = True, save_afc_no_iti: bool = True) -> pd.DataFrame:
    """
    Main function to find feasible itineraries for all passengers and save results.
    This function will generate two dataframes:
        1. AFC_feas_iti_not_found.pkl: records of passengers without feasible itineraries.
            (Could be empty, if empty, not saved.)
        2. feas_iti.pkl: feasible itineraries for all passengers. (with the returned df structure)

    NOTE: The full execution typically takes around **10 minutes**, depending on system performance.

    :return: pd.DataFrame containing feasible itineraries.
        columns: ['rid', 'iti_id', 'path_id','seg_id', 'train_id', 'board_ts', 'alight_ts']
    """
    from src.globals import get_k_pv_dict, get_afc
    afc = get_afc()
    k_pv_dic = get_k_pv_dict()

    data = []
    for rid, uid1, ts1, uid2, ts2 in tqdm(afc, total=afc.shape[0], desc="Finding feasible itineraries"):
        k_pv = k_pv_dic[(uid1, uid2)]
        iti_list = find_feas_iti(k_pv, ts1, ts2)
        for iti_id, itinerary in enumerate(iti_list, start=1):
            path_id = itinerary[0]
            for seg_id, (train_id, board_ts, alight_ts) in enumerate(itinerary[1:], start=1):
                data.append([rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts])

    def _save_rids_not_found() -> pd.DataFrame | None:
        """Helper function to save records of passengers without feasible itineraries."""
        if not data:
            print("All passengers have feasible itineraries.")
            return None
        rids_not_found = afc[~np.isin(afc[:, 0], [seg[0] for seg in data])]
        df_rids_not_found = pd.DataFrame(
            rids_not_found,
            columns=['rid', 'uid1', 'ts1', 'uid2', 'ts2']
        )
        print(f"Not found feasible itinerary: {len(rids_not_found)} passengers.")
        return df_rids_not_found

    if save_afc_no_iti:  # save rids without feasible itineraries
        iti_not_found = _save_rids_not_found()
        if iti_not_found is not None:
            save_(fn=config.CONFIG["results"]["AFC_no_iti"], data=iti_not_found, auto_index_on=False)

    data = np.array(data, dtype=np.int32)

    # save feasible itineraries to file with pandas DataFrame
    df = pd.DataFrame(
        data,
        columns=['rid', 'iti_id', 'path_id', 'seg_id', 'train_id', 'board_ts', 'alight_ts'])
    df = df.astype({
        'rid': 'int32',
        'iti_id': 'int32',
        'path_id': 'int32',
        'seg_id': 'int8',
        'train_id': 'category',
        'board_ts': 'int32',
        'alight_ts': 'int32'
    })

    if save_feas_iti:
        save_(fn=config.CONFIG["results"]["feas_iti"], data=df, auto_index_on=False)

    return df


def process_afc_chunk(chunk: np.ndarray, k_pv_dict: dict) -> list[list]:
    results = []
    for rid, uid1, ts1, uid2, ts2 in chunk:
        k_pv = k_pv_dict[(uid1, uid2)]
        iti_list = find_feas_iti(k_pv, ts1, ts2)
        for iti_id, itinerary in enumerate(iti_list, start=1):
            path_id = itinerary[0]
            for seg_id, (train_id, board_ts, alight_ts) in enumerate(itinerary[1:], start=1):
                results.append([rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts])
    return results


def find_feas_iti_all_parallel(save_feas_iti: bool = True, save_afc_no_iti: bool = True,
                               n_jobs: int = -1, chunk_size: int = 100000) -> pd.DataFrame:
    """
    TODO: This function is not tested yet. Please use find_feas_iti_all instead. (2025-04-21) Can't display progress bar.
    Parallel version: Find feasible itineraries for all passengers and save results.

    :param save_feas_iti: If True, save feasible itineraries to file.
    :param save_afc_no_iti: If True, save records of passengers without feasible itineraries.
    :param n_jobs: Number of parallel jobs. If -1, use all available CPU cores. Default is -1.
    :param chunk_size: Number of rids to process in each chunk. Default is 100,000.

    :return: pd.DataFrame with ['rid', 'iti_id', 'path_id','seg_id', 'train_id', 'board_ts', 'alight_ts']
    """
    from src.globals import get_afc, build_k_pv_dic
    afc = get_afc()
    k_pv_dict = build_k_pv_dic()

    chunks = [afc[i:i + chunk_size] for i in range(0, len(afc), chunk_size)]

    print(f"[INFO] Start finding feasible itineraries using {n_jobs} threads with chunk size {chunk_size}...")

    partial_process_afc_chunk = partial(process_afc_chunk, k_pv_dict=k_pv_dict)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(partial_process_afc_chunk)(chunk) for chunk in chunks
    )
    data = [row for group in results for row in group]

    def _save_rids_not_found() -> pd.DataFrame | None:
        if not data:
            print("All passengers have no feasible itineraries.")
            return pd.DataFrame(afc, columns=['rid', 'uid1', 'ts1', 'uid2', 'ts2'])
        found_rids = np.array([row[0] for row in data])
        not_found_mask = ~np.isin(afc[:, 0], found_rids)
        rids_not_found = afc[not_found_mask]
        if rids_not_found.shape[0] > 0:
            print(f"[INFO] Not found feasible itinerary for {rids_not_found.shape[0]} passengers.")
            return pd.DataFrame(rids_not_found, columns=['rid', 'uid1', 'ts1', 'uid2', 'ts2'])
        return None

    if save_afc_no_iti:
        df_nf = _save_rids_not_found()
        if df_nf is not None:
            save_(fn=config.CONFIG["results"]["AFC_no_iti"], data=df_nf, auto_index_on=False)

    # Convert to final dataframe
    df = pd.DataFrame(
        data,
        columns=['rid', 'iti_id', 'path_id', 'seg_id', 'train_id', 'board_ts', 'alight_ts']
    ).astype({
        'rid': 'int32',
        'iti_id': 'int32',
        'path_id': 'int32',
        'seg_id': 'int8',
        'train_id': 'category',
        'board_ts': 'int32',
        'alight_ts': 'int32'
    })

    if save_feas_iti:
        save_(fn=config.CONFIG["results"]["feas_iti"], data=df, auto_index_on=False)

    return df
