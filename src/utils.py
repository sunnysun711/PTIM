import time

import pandas as pd

DATA_DIR = "data"


def execution_timer(func):
    def wrapper(*args, **kw):
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
def read_data(fn: str = "AFC") -> pd.DataFrame:
    df = pd.read_pickle(f"{DATA_DIR}//cd//{fn}.pkl")
    if fn == "AFC":
        df = df.drop(columns=["STATION1_NID", "STATION2_NID", "STATION1_TIME", "STATION2_TIME"])
    elif fn == "STA":
        df = df.drop(columns=["STATION_NAME", "STATION_NAME_E"])
    elif fn == "TT":
        df = df.drop(
            columns=["TRAIN_NUMBER", "STATION_NAME", "O_STATION", "T_STATION", "ARRIVE_TIME", "DEPARTURE_TIME",
                     "TRAIN_ID_old"])
    else:
        raise ValueError(f"fn should be 'AFC', 'TT' or 'STA'.")
    return df


def tstr2ts(t: str) -> int:
    """convert from time string "08:05" to timestamp 29100."""
    return int(int(t[:2]) * 3600 + int(t[-2:]) * 60)


def ts2tstr(ts: int, include_seconds: bool = False) -> str:
    """convert from timestamp 29580 to time string "08:13"."""
    s = ts % 60  # seconds
    h = ts // 3600  # hour
    m = ts // 60 % 60  # minutes
    if include_seconds:
        return f"{h:02}:{m:02}:{s:02}"
    else:
        return f"{h:02}:{m:02}"


if __name__ == '__main__':
    print(ts2tstr(75081, False))
