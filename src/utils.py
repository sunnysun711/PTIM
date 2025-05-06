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
import contextlib
import functools
import json
import os
import sys
import time

import joblib
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
        print(
            f"[INFO] {func.__name__}({args}, {kw}) executing at {start_str}.")

        res = func(*args, **kw)  # 执行函数

        b = time.time()
        finish_str = time.strftime("%m-%d %H:%M:%S", time.localtime(b))
        print(
            f"[INFO] {func.__name__}({args}, {kw}) executed in  {finish_str}. ({start_str} -> {b - a:.4f}s )")
        return res

    return wrapper


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Refer to: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def split_fn_index_ext(full_fn: str) -> tuple[str, str, str]:
    """
    Split a filename into its base name, index, and extension.

    :param full_fn: The full filename including the base name, index, and extension.
    :type full_fn: str

    :return: A tuple containing the base name, index, and extension.
    :rtype: tuple[str, str, str]

    Example:
        >>> split_fn_index_ext("egress_times_2.pkl") 
        ('egress_times', '_2', '.pkl')

        >>> split_fn_index_ext("iti/AFC_no_iti.pkl") 
        ('iti/AFC_no_iti', '', '.pkl')
    """
    parts = full_fn.rsplit(".", 1)
    fn = parts[0]
    ext = "." + parts[1] if len(parts) > 1 else ".pkl"
    index = "_" + fn.split("_")[-1] if fn.split("_")[-1].isdigit() else ""
    fn = fn[:len(fn) - len(index)] if index else fn
    return fn, index, ext


def get_folder_and_subfolder(fn: str, ext: str) -> tuple[str, str]:
    """
    Determines the folder and subfolder based on the file name.

    :param fn: The base file name (without extension).
    :type fn: str
    :param ext: The file extension.
    :type ext: str

    :returns: A tuple containing the folder and subfolder.
    :rtype: tuple[str, str]

    Example:
    --------
    >>> get_folder_and_subfolder("path", ".pkl")
    ('results', 'network')

    >>> get_folder_and_subfolder("AFC", ".pkl")
    ('data', '')
    """
    if fn + ext in config.CONFIG['data']:
        return config.CONFIG['data_folder'], ""
    elif fn + ext in config.CONFIG['results'].values():
        return config.CONFIG['results_folder'], config.determine_results_subfolder(fn + ext)
    else:
        raise ValueError(f"Unknown file: {fn + ext}")


def get_file_path(fn: str, latest_: bool = False) -> str:
    """
    Determines the file path based on the file name (fn) and the configuration. (no index)

    :param fn: The name of the file. The function checks whether the file is a part of the
        data or results sections in the config and constructs the appropriate path.
    :type fn: str
    :param latest_: (optional) If True, return the path for the latest file version.
    :type latest_: bool, optional

    :returns: The full file path including the folder, subfolder, and file name.
    :rtype: str

    :raises ValueError: If the file name is not found in the predefined list of data or results in the configuration.

    Example:
    --------
    >>> get_file_path("data_file", latest_=True)
    '/data/results/latest/data_file.csv'
    """
    fn, index, ext = split_fn_index_ext(fn)
    folder, subfolder = get_folder_and_subfolder(fn, ext)
    if latest_:
        index = f"_{get_latest_file_index(fn+ext, folder=folder, subfolder=subfolder, get_next=False)}"

    return os.path.join(folder, subfolder, fn + index + ext)


def get_latest_file_index(fn: str, folder: str = "", subfolder: str = "", get_next: bool = False) -> int:
    """
    Get the latest file index for versioned files.

    :param fn: The base file name (without extension).
    :type fn: str
    :param folder: The folder path.
    :type folder: str, optional
    :param subfolder: The subfolder path.
    :type subfolder: str, optional
    :param get_next: If True, return the next available index. If False, return the latest index.
    :type get_next: bool, optional

    :returns: The index of the latest file.
    :rtype: int

    Example:
    --------
    >>> get_latest_file_index("etd", folder="results", subfolder="egress", get_next=True)
    5
    """
    base_fn, index, ext = split_fn_index_ext(fn)
    for i in range(1, 10001):
        if not os.path.exists(os.path.join(folder, subfolder, f"{base_fn}_{i}{ext}")):
            if get_next:
                return i
            return i - 1
    else:
        raise RuntimeError(
            f"Could not find latest file index: all {base_fn}_1{ext} to _10000{ext} exist.")


@execution_timer
def read_(
    fn: str = "AFC",
    show_timer: bool = False,
    drop_cols: bool = True,
    latest_: bool = False, 
    quiet_mode: bool = False,
) -> pd.DataFrame | dict | list:
    """
    Reads a file from a specified folder based on its name and extension. 

    The function handles different file formats such as .csv, .parquet, .pkl, 
    and .json, and returns the corresponding data as a pandas DataFrame, 
    dictionary, or list.

    :param fn: The name of the file to be read. 

        The file name can be specified without an extension, and the function 
        will automatically append the appropriate extension (.pkl, .csv, 
        .parquet, or .json) based on the file type.
    :type fn: str, optional, default="AFC"

    :param show_timer: If set to True, the function will display a timer 
        showing how long the read operation takes. (Currently unused).
    :type show_timer: bool, optional, default=False

    :param drop_cols: If set to True, the function will drop certain columns 
        from the DataFrame based on the file name, such as "STATION1_NID", 
        "STATION2_NID", and others for specific files like "AFC.pkl" or "STA.pkl".
    :type drop_cols: bool, optional, default=True

    :param latest_: If set to True, the function will read the latest version of 
        the file based on the file name. 

        The latest version is determined by checking for files with the same base 
        name but with an index suffix (_1, _2, etc.).

        If no latest version is found, the function will raise a FileNotFoundError.
    :type latest_: bool, optional, default=False
    
    :param quiet_mode: If set to True, the function will not print any messages
        to the console.
    :type quiet_mode: bool, optional, default=False

    :returns: The function returns a pandas DataFrame, dictionary, or list, depending on the file format:

        - DataFrame for .csv, .parquet, and .pkl files.
        - Dictionary for .json files.
    :rtype: pd.DataFrame | dict | list

    :raises ValueError: If the file name is not found in the predefined list of data or results in the configuration, or if
        the file extension is unrecognized.

    Example:
    --------
    >>> read_(fn="AFC", show_timer=True, drop_cols=True)
    [INFO] Reading file: data\AFC.pkl

    >>> read_("assigned.pkl", latest_=True)
    [INFO] Reading file: results\trajectory\assigned_1.pkl

    >>> read_("assigned_1", latest_=False)  # Explicitly specify the file name with index but set latest_=False
    [INFO] Reading file: results\trajectory\assigned_1.pkl

    >>> read_("platform.json")
    [INFO] Reading file: data\platform.json
    """
    fp = get_file_path(fn, latest_)
    if not quiet_mode:
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
            df = {int(key) if key.isdigit()
                  else key: value for key, value in df.items()}
    else:
        raise ValueError(f"Unknown file extension: {fp}")

    if drop_cols:
        if fn == "AFC":
            df = df.drop(
                columns=["STATION1_NID", "STATION2_NID", "STATION1_TIME", "STATION2_TIME"])
        elif fn == "STA":
            df = df.drop(columns=["STATION_NAME", "STATION_NAME_E"])
        elif fn == "TT":
            df = df.drop(
                columns=["TRAIN_NUMBER", "STATION_NAME", "O_STATION", "T_STATION", "ARRIVE_TIME", "DEPARTURE_TIME",
                         "TRAIN_ID_old"])
    return df


def save_(fn: str, data: pd.DataFrame, auto_index_on: bool = False, verbose: bool = True) -> None:
    """
    Saves a DataFrame to a .pkl/.csv/.parquet file.

    :param fn: The name of the file. The function will save the data to the appropriate directory based on the configuration.
    :type fn: str

    :param data: The pandas DataFrame to be saved.
    :type data: pd.DataFrame

    :param auto_index_on: If set to True, the function will automatically append an index to the file name
        to avoid overwriting existing files.
    :type auto_index_on: bool, optional, default=False

    :param verbose: If True, print a sample and data.info() before saving.
    :type verbose: bool, optional, default=True

    :returns: None

    :raises ValueError: If the file extension is not in `.pkl`, `.csv`, `.parquet`.

    Example:
    --------
    >>> save_("my_data", my_dataframe)
    >>> save_(config.CONFIG["results"]["egress_times"], egress_times_df, auto_index_on=True, verbose=False)
    """
    if len(fn.split(".")) == 1:  # No extension
        fn = f"{fn}.pkl"
    if fn.endswith(".pkl"):
        saving_method = data.to_pickle
    elif fn.endswith(".csv"):
        def saving_method(x): return data.to_csv(
            x, index=data.index.name is not None)  # Keep index name if it exists
    elif fn.endswith(".parquet"):
        saving_method = data.to_parquet
    else:
        raise ValueError(
            "Only .pkl, .csv, .parquet files can be saved using this method.")

    fp = get_file_path(fn)  # Use the same function to get the correct path
    if auto_index_on:
        fp = fp.split(
            ".")[0] + f"_{get_latest_file_index(fp, get_next=True)}." + fp.split(".")[-1]

    if verbose:
        print(data.sample(n=min(10, len(data))))
        data.info()

    saving_method(fp)
    print(f"\033[1;91m[INFO] File saved to: {fp}\033[0m")  # bold red
    return


@execution_timer
def read_all(fn: str, show_timer: bool = False) -> pd.DataFrame:
    """
    Reads all versioned files (e.g., base_1.pkl ~ base_10000.pkl) and concatenates them into a single DataFrame.

    :param fn: Prefix of the file.
    :type fn: str

    :param show_timer: Whether to show timing information.
    :type show_timer: bool

    :return: Concatenated DataFrame.
    :rtype: pd.DataFrame
    """
    dfs = []
    fp = get_file_path(fn)
    latest_id = get_latest_file_index(fp, get_next=False)
    for i in range(1, latest_id + 1):
        file_path = fp.split(".")[0] + f"_{i}." + fp.split(".")[-1]
        if os.path.exists(file_path):
            dfs.append(pd.read_pickle(file_path))
        else:
            break  # stop when gap is hit
    if not dfs:
        raise FileNotFoundError(f"No versioned files found for {fn}")
    print(f"[INFO] Reading {len(dfs)} versioned files: {fp}")
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


class AssignLogger:
    """
    Assignment Logger, writes to both terminal and log file.
    
    Log file is named by config.CONFIG["results"]["assignment_log"].
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.fn = config.CONFIG["results"]["assignment_log"]
        log_file_path = get_file_path(self.fn)
        self.log = open(log_file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
