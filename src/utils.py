import json
import time

import pandas as pd

DATA_DIR = "data//cd"


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
def read_data(fn: str = "AFC", show_timer: bool = False) -> pd.DataFrame:
    """Read data file and drop specific columns based on file name."""
    df = pd.read_pickle(f"{DATA_DIR}//{fn}.pkl")
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
