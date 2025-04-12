"""
This utility module provides reusable functions for file handling, timing, and data conversion.
Includes decorators for timing and auto-saving, along with utility functions for reading and converting data.

Key Functions:
1. read_data: Load various file types with optional column pruning
2. file_saver / file_auto_index_saver: Decorators for saving DataFrames with or without versioning
3. read_data_latest / read_data_all: Load the latest or all versioned data files
4. ts2tstr / tstr2ts: Timestamp and string time format conversion

Dependencies:
- pandas, os, json, time
"""
import functools
import json
import os
import time

import pandas as pd

from src import DATA_DIR


def execution_timer(func):
    """Function execution timer decorator, optionally display timing information."""

    def wrapper(*args, **kw):
        if "show_timer" in kw and not kw['show_timer']:  # don't print anything
            res = func(*args, **kw)  # 执行函数
            return res

        a = time.time()
        start_str = time.strftime("%m-%d %H:%M:%S", time.localtime(a))
        print(f"[INFO] {func.__name__}({args}, {kw}) executing at {start_str}.")

        res = func(*args, **kw)  # 执行函数

        b = time.time()
        finish_str = time.strftime("%m-%d %H:%M:%S", time.localtime(b))
        print(f"[INFO] {func.__name__}({args}, {kw}) executed in  {finish_str}. ({start_str} -> {b - a:.4f}s )")
        return res

    return wrapper


def file_saver(func):
    """File saver decorator that automatically saves function results to pickle files.

    Decorator that wraps a function returning a pandas DataFrame/object and:
    - Optionally saves the result to a .pkl file when 'save_fn' parameter is provided
    - Provides sample output and memory usage info before saving
    - Handles empty/None results gracefully

    Usage:
        @file_saver
        def my_func(..., save_fn=None):
            # returns a DataFrame
            return df

        >>> my_func(save_fn="my_data")
        # return the result of my_func("my_data") and
        # saves to my_data.pkl

        >>> my_func("my_data")
        # this will not save the data.
        # it will just return the result of my_func("my_data")
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the file name from the save_fn parameter
        save_fn = kwargs.get('save_fn', None)

        # Call the original function to generate the dataframe
        res = func(*args, **kwargs)

        if save_fn:
            if res is None:
                print(f"[INFO] {save_fn} is empty. Skipping saving.")
                return res
            # display saving dataframe
            print(res.sample(n=min(10, len(res))))
            res.info()

            full_fn = fr"{DATA_DIR}/{save_fn}.pkl"
            res.to_pickle(full_fn)
            print(f"[INFO] {full_fn} saved.")

        return res

    return wrapper


def file_auto_index_saver(func):
    """File saver with auto-incremented filename suffix (_1, _2, ...), using same API as file_saver.

    Usage:
        @file_auto_index_saver
        def func(..., save_fn=None): ...

        >>> my_func(save_fn="my_data")
        # return the result of my_func("my_data") and
        # saves to my_data_1.pkl / my_data_2.pkl / ...
        # if my_data_1.pkl exists, my_data_2.pkl will be used.

        >>> my_func("my_data")
        # this will not save the data.
        # it will just return the result of my_func("my_data")
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        save_fn = kwargs.get('save_fn', None)

        if save_fn:
            for i in range(1, 10001):
                indexed_save_fn = f"{save_fn}_{i}"
                file_path = f"{DATA_DIR}/{indexed_save_fn}.pkl"
                if not os.path.exists(file_path):
                    kwargs["save_fn"] = indexed_save_fn
                    break
            else:
                raise RuntimeError(f"Could not save: all {save_fn}_1.pkl to _10000.pkl exist.")

        # Call original function via file_saver wrapper
        return file_saver(func)(*args, **kwargs)

    return wrapper


@execution_timer  # ~ 0.1433 seconds for AFC
def read_data(fn: str = "AFC", show_timer: bool = False, drop_cols: bool = True) -> pd.DataFrame | dict | list:
    """Read data file and drop specific columns based on file name."""
    if fn.endswith(".csv"):
        df = pd.read_csv(f"{DATA_DIR}//{fn}")
    elif fn.endswith(".parquet"):
        df = pd.read_parquet(f"{DATA_DIR}//{fn}")
    elif fn.endswith(".pkl"):
        df = pd.read_pickle(f"{DATA_DIR}//{fn}")
    elif fn.endswith(".json"):
        with open(f"{DATA_DIR}//{fn}", "r") as f:
            df = json.load(f)
    else:
        df = pd.read_pickle(f"{DATA_DIR}//{fn}.pkl")

    if drop_cols:
        if fn == "AFC":
            df = df.drop(columns=["STATION1_NID", "STATION2_NID", "STATION1_TIME", "STATION2_TIME"])
        elif fn == "STA":
            df = df.drop(columns=["STATION_NAME", "STATION_NAME_E"])
        elif fn == "TT":
            df = df.drop(
                columns=["TRAIN_NUMBER", "STATION_NAME", "O_STATION", "T_STATION", "ARRIVE_TIME", "DEPARTURE_TIME",
                         "TRAIN_ID_old"])
    return df


def read_platform_exceptions() -> dict[int, list[list[int]]]:
    """
    Read platform.json file and convert keys (uid) to integers.

    Keys are station uid, type int.

    Values are list of connected platform node id.

    :return: dict[uid, [[node_id, node_id], [node_id]]]
    """
    file = f"{DATA_DIR}//platform.json"
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
        # convert from str to int for station uid
        data = {int(key) if key.isdigit() else key: value for key, value in data.items()}
    return data


def find_data_latest(fn: str) -> str:
    """
    Find the latest versioned file (e.g., base_1.pkl ~ base_10000.pkl).
    :param fn: Prefix of the file.
    :return: Latest file path.
    """
    for i in reversed(range(1, 10001)):
        file_path = f"{DATA_DIR}/{fn}_{i}.pkl"
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"No versioned file found for {fn} in range 1-10000.")


@execution_timer
def read_data_latest(fn: str, show_timer: bool = False) -> pd.DataFrame:
    """
    Load the latest versioned file (e.g., base_1.pkl ~ base_10000.pkl).

    :param fn: Prefix of the file.
    :return: Loaded DataFrame.
    """
    file_path = find_data_latest(fn)
    print(f"[INFO] Loading latest: {file_path}")
    return pd.read_pickle(file_path)


@execution_timer
def read_data_all(fn: str, show_timer: bool = False) -> pd.DataFrame:
    """
    Load and concatenate all versioned files (e.g., base_1.pkl ~ base_10000.pkl).

    :param fn: Prefix of the file.
    :param show_timer: Whether to show timing information.
    :return: Concatenated DataFrame.
    """
    dfs = []
    for i in range(1, 10001):
        file_path = f"{DATA_DIR}/{fn}_{i}.pkl"
        if os.path.exists(file_path):
            dfs.append(pd.read_pickle(file_path))
        else:
            break  # stop when gap is hit
    if not dfs:
        raise FileNotFoundError(f"No versioned files found for {fn}")
    print(f"[INFO] Loaded {len(dfs)} versioned files for {fn}")
    return pd.concat(dfs, ignore_index=True)


def tstr2ts(t: str) -> int:
    """convert from time string "08:05" to timestamp 29100."""
    return int(int(t[:2]) * 3600 + int(t[-2:]) * 60)


def ts2tstr(ts: int, include_seconds: bool = False) -> str:
    """convert from timestamp 29580 to time string "08:13", optionally include seconds."""
    s = ts % 60  # seconds
    h = ts // 3600  # hour
    m = ts // 60 % 60  # minutes
    if include_seconds:
        return f"{h:02}:{m:02}:{s:02}"
    else:
        return f"{h:02}:{m:02}"


if __name__ == '__main__':
    print(ts2tstr(75081, False))
