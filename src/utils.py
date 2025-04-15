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

from src import config


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


@execution_timer  # ~ 0.1433 seconds for AFC
def read_data(fn: str = "AFC", show_timer: bool = False, drop_cols: bool = True) -> pd.DataFrame | dict | list:
    """Read data file and drop specific columns based on file name."""
    if fn.endswith(".csv"):
        df = pd.read_csv(f"{config.CONFIG['data_folder']}//{fn}")
    elif fn.endswith(".parquet"):
        df = pd.read_parquet(f"{config.CONFIG['data_folder']}//{fn}")
    elif fn.endswith(".pkl"):
        df = pd.read_pickle(f"{config.CONFIG['data_folder']}//{fn}")
    elif fn.endswith(".json"):
        with open(f"{config.CONFIG['data_folder']}//{fn}", "r") as f:
            df = json.load(f)
    else:
        df = pd.read_pickle(f"{config.CONFIG['data_folder']}//{fn}.pkl")

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


def split_fn_index_ext(full_fn: str) -> tuple[str, str, str]:
    """
    Split a filename into its base name, index, and extension.
    Parameters:
    -----------
    full_fn : str
        The full filename including the base name, index, and extension.
    Returns:
    --------
    tuple[str, str, str]
        A tuple containing the base name, index, and extension.
        Example:
            egress_times_2.pkl -> ('egress_times', '_2', '.pkl')
            iti/AFC_no_iti.pkl -> ('iti/AFC_no_iti', '', '.pkl')
    """
    ext = "." + full_fn.split(".")[-1] if len(full_fn.split(".")) > 1 else ".pkl"
    fn = full_fn.split(".")[0]
    index = "_" + fn.split("_")[-1] if fn.split("_")[-1].isdigit() else ""
    fn = fn[:-len(index)] if index else fn
    return fn, index, ext


def get_file_path(fn: str) -> str:
    """
    Determines the file path based on the file name (fn) and the configuration. (no index)

    Parameters:
    -----------
    fn : str
        The name of the file. The function checks whether the file is a part of the
        data or results sections in the config and constructs the appropriate path.

    Returns:
    --------
    str
        The full file path including the folder, subfolder, and file name.

    Raises:
    -------
    ValueError
        If the file name is not found in the predefined list of data or results in the configuration.
    """
    fn, index, ext = split_fn_index_ext(fn)
    if fn + ext in config.CONFIG['data']:
        folder = config.CONFIG['data_folder']
        subfolder = ""
    elif fn + ext in config.CONFIG['results'].values():
        folder = config.CONFIG['results_folder']
        subfolder = determine_results_subfolder(fn + ext)
    else:
        raise ValueError(f"Unknown file: {fn + index + ext}")

    fp = os.path.join(folder, subfolder, fn + index + ext)
    return fp


def determine_results_subfolder(fn: str) -> str:
    """
    Determines the subfolder based on the file name (fn) and the configuration.
    Parameters:
    -----------
    fn : str
        The name of the file. The function checks whether the file is a part of the
        data or results sections in the config and constructs the appropriate path.
    Returns:
    --------
    str
        The full file path including the folder, subfolder, and file name.
    Raises:
    -------
    ValueError
        If the file name is not found in the predefined list of data or results in the configuration.
    """
    if fn in [config.CONFIG["results"]["node"], config.CONFIG["results"]["link"]]:
        return config.CONFIG["results_subfolder"]["network"]
    elif fn in [config.CONFIG["results"]["path"], config.CONFIG["results"]["pathvia"]]:
        return config.CONFIG["results_subfolder"]["path"]
    elif fn in [config.CONFIG["results"]["feas_iti"], config.CONFIG["results"]["AFC_no_iti"]]:
        return config.CONFIG["results_subfolder"]["itinerary"]
    elif fn in [config.CONFIG["results"]["egress_times"]]:
        return config.CONFIG["results_subfolder"]["egress"]
    elif fn in [config.CONFIG["results"]["assigned"], config.CONFIG["results"]["left"],
                config.CONFIG["results"]["stashed"]]:
        return config.CONFIG["results_subfolder"]["trajectory"]
    else:
        raise ValueError(f"Unknown file to determine results subfolder: {fn}")


def get_latest_file_index(fn: str, get_next: bool = False) -> int:
    fp = get_file_path(fn)
    base_fp, index, ext = split_fn_index_ext(full_fn=fp)
    for i in range(1, 10001):
        if not os.path.exists(base_fp + f"_{i}" + ext):
            if get_next:
                return i
            return i - 1
    else:
        raise RuntimeError(f"Could not find latest file index: all {base_fp}_1.pkl to _10000.pkl exist.")


@execution_timer
def read_(fn: str = "AFC", show_timer: bool = False, drop_cols: bool = True,
          latest_: bool = False) -> pd.DataFrame | dict | list:
    """
    Reads a file from a specified folder based on its name and extension. The function handles different file
    formats such as .csv, .parquet, .pkl, and .json, and returns the corresponding data as a pandas DataFrame,
    dictionary, or list.

    Parameters:
    -----------
    fn : str, optional, default="AFC"
        The name of the file to be read. The file name can be specified without an extension, and the function
        will automatically append the appropriate extension (.pkl, .csv, .parquet, or .json) based on the file type.

    show_timer : bool, optional, default=False
        If set to True, the function will display a timer showing how long the read operation takes. (Currently unused).

    drop_cols : bool, optional, default=True
        If set to True, the function will drop certain columns from the DataFrame based on the file name, such as
        "STATION1_NID", "STATION2_NID", and others for specific files like "AFC.pkl" or "STA.pkl".

    latest_ : bool, optional, default=False
        If set to True, the function will read the latest version of the file based on the file name. The latest
        version is determined by checking for files with the same base name but with an index suffix (_1, _2, etc.).
        If no latest version is found, the function will raise a FileNotFoundError.

    Returns:
    --------
    pd.DataFrame | dict | list
        The function returns a pandas DataFrame, dictionary, or list, depending on the file format:
        - DataFrame for .csv, .parquet, and .pkl files.
        - Dictionary for .json files.

    Raises:
    -------
    ValueError
        If the file name is not found in the predefined list of data or results in the configuration, or if
        the file extension is unrecognized.

    Example:
    --------
    >>> read_(fn="AFC", show_timer=True, drop_cols=True)
    [INFO] Reading file: data\AFC.pkl

    >>> from src import config
    >>> config.load_config()
    >>> read_(config.CONFIG["results"]["assigned"], latest_=True)
    [INFO] Reading file: results\trajectory\assigned_1.pkl

    >>> read_("assigned_1", latest_=False)  # Explicitly specify the file name with index but set latest_=False
    [INFO] Reading file: results\trajectory\assigned_1.pkl

    >>> read_("platform.json")
    [INFO] Reading file: data\platform.json
    """
    fp = get_file_path(fn)
    if latest_:
        fp = fp.split(".")[0] + f"_{get_latest_file_index(fn, get_next=False)}." + fp.split(".")[-1]
    print("[INFO] Reading file:", fp)

    if fp.endswith(".csv"):
        df = pd.read_csv(fp)
    elif fp.endswith(".parquet"):
        df = pd.read_parquet(fp)
    elif fp.endswith(".pkl"):
        df = pd.read_pickle(fp)
    elif fp.endswith(".json"):
        with open(fp, "r", encoding="utf-8") as f:
            df = json.load(f)
            df = {int(key) if key.isdigit() else key: value for key, value in df.items()}
    else:
        raise ValueError(f"Unknown file extension: {fp}")

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


def save_(fn: str, data: pd.DataFrame, auto_index_on: bool = False) -> None:
    """
    Saves a DataFrame to a .pkl file.

    Parameters:
    -----------
    fn : str
        The name of the file. This should not include the `.pkl` extension. The function will save the data to the
        appropriate directory based on the configuration.

    data : pd.DataFrame
        The pandas DataFrame to be saved.

    auto_index_on : bool, optional, default=False
        If set to True, the function will automatically append an index to the file name to avoid overwriting existing
        files. The index will be appended as `_1`, `_2`, etc., up to `_10000`. If no available index is found, the
        function will raise a RuntimeError.

    Example:
    --------
    save_("my_data", my_dataframe)

    Raises:
    -------
    ValueError
        If the file extension is not `.pkl`.
    """
    if len(fn.split(".")) == 1:  # No extension
        fn = f"{fn}.pkl"
    if not fn.endswith(".pkl"):
        raise ValueError("Only .pkl files can be saved using this method.")

    fp = get_file_path(fn)  # Use the same function to get the correct path
    if auto_index_on:
        fp = fp.split(".")[0] + f"_{get_latest_file_index(fn, get_next=True)}." + fp.split(".")[-1]

    # display saving dataframe
    print(data.sample(n=min(10, len(data))))
    data.info()

    data.to_pickle(fp)  # Save the DataFrame as a .pkl file
    print(f"[INFO] File saved to: {fp}")
    return


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
        file_path = f"{config.CONFIG['data_folder']}/{fn}_{i}.pkl"
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
