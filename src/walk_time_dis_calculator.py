"""
walk_time_dis_calculator.py

A module for efficient and scalable calculation of walking time distributions
in metro systems, including entry, egress, and transfer scenarios.

This module defines the `WalkTimeDisCalculator` class, which provides methods
to compute probability distribution values (PDF/CDF) for walking times, given
a passenger's travel path and segment. The distributions are preloaded and
cached for fast lookup, supporting both scalar and batch queries.

Key Features:
-------------
- Calculates:
    - Egress time probability density function (PDF)
    - Entry time cumulative distribution function (CDF)
    - Transfer time cumulative distribution function (CDF)
- Uses NumPy arrays and lookup tables for high-performance vectorized access
- Supports platform-specific data exceptions (e.g., TaiPingYuan)

Main Class:
-----------
- WalkTimeDisCalculator

    Methods:
    - get_egress_dis(path_id, times)
    - get_entry_dis(path_id, t_start, t_end)
    - get_transfer_dis(path_id, seg_id, t_start, t_end)

Data Requirements:
------------------
- ETD (Egress Time Distribution): ndarray, shape (N, 4)
    Each row: [pp_id, x, pdf, cdf]
- TTD (Transfer Time Distribution): ndarray, shape (M, 4)
    Each row: [pp_id_min, pp_id_max, x, cdf]
- Platform Mapping: get_platform()
- K-shortest path info: get_k_pv()
- Transfer segment info: get_path_seg_to_pp_ids()

Dependencies:
-------------
- numpy
- pandas
- src.config
- src.globals:
    - get_etd
    - get_ttd
    - get_k_pv
    - get_platform
- src.walk_time_filter:
    - get_path_seg_to_pp_ids

Usage Example:
--------------
>>> from walk_time_dis_calculator import WalkTimeDisCalculator
>>> calculator = WalkTimeDisCalculator()
>>> pdf = calculator.get_egress_dis(path_id=1001, times=[10, 20, 30])
>>> cdf = calculator.get_entry_dis(path_id=1001, t_start=5, t_end=25)
>>> cdf_transfer = calculator.get_transfer_dis(path_id=1001, seg_id=2, t_start=10, t_end=35)
"""

import numpy as np
import pandas as pd

from src import config
from src.globals import get_etd, get_ttd, get_k_pv, get_platform
from src.walk_time_filter import get_path_seg_to_pp_ids


class WalkTimeDisCalculator:
    def __init__(
        self,
        etd: np.ndarray = None,
        ttd: np.ndarray = None,
    ):
        """
        Walk Time Distribution Calculator

        Calculates walking time distributions for:
            - Egress (PDF)
            - Entry (CDF)
            - Transfer (CDF)
        """
        # Load ETD and TTD
        self.etd: np.ndarray = etd if etd is not None else get_etd()
        self.ttd: np.ndarray = ttd if ttd is not None else get_ttd()

        # Initialize mappings
        self._load_path_entry_egress_mapping()
        self._load_transfer_mapping()

        # Build lookup tables
        self._build_pp_id_lookup_table(2, "pp_id2pdf_table")  # Egress PDF
        self._build_pp_id_lookup_table(3, "pp_id2cdf_table")  # Entry CDF
        self._build_transfer_lookup_table()  # Transfer CDF

    def _load_path_entry_egress_mapping(self):
        """Load path_id to entry/egress pp_id mappings."""
        df_k_pv = pd.DataFrame(
            get_k_pv()[:, :-2],
            columns=["path_id", "pv_id", "node_id1", "node_id2", "link_type"],
        )

        node_id_to_pp_id = {
            node_id: pp_id for node_id, pp_id in get_platform()[:, [1, 0]]
        }

        df_entry = df_k_pv[df_k_pv["pv_id"] == 1].copy()
        df_egress = df_k_pv.iloc[df_entry.index - 1].copy()

        assert (
            df_entry["link_type"] == "entry"
        ).all(), "df_entry contains incorrect link types."
        assert (
            df_egress["link_type"] == "egress"
        ).all(), "df_egress contains incorrect link types."

        df_entry["pp_id"] = df_entry["node_id2"].map(node_id_to_pp_id)
        df_egress["pp_id"] = df_egress["node_id1"].map(node_id_to_pp_id)

        self.path_id2entry_pp_id: dict[int, int] = df_entry.set_index("path_id")[
            "pp_id"
        ].to_dict()
        self.path_id2egress_pp_id: dict[int, int] = df_egress.set_index("path_id")[
            "pp_id"
        ].to_dict()

    def _load_transfer_mapping(self):
        """Load path-segment to transfer pp_id_min and pp_id_max mappings."""
        df = get_path_seg_to_pp_ids()
        df["pp_id_min"] = df[["pp_id1", "pp_id2"]].min(axis=1)
        df["pp_id_max"] = df[["pp_id1", "pp_id2"]].max(axis=1)
        self.path_seg2pp_id_mima: pd.DataFrame = df.drop(columns=["pp_id1", "pp_id2"])

    def _build_pp_id_lookup_table(
        self,
        column_idx: int,
        attr_name: str,
    ):
        """
        General function to build lookup tables for pp_id -> distribution values.

        Args:
            column_idx (int): The column index to extract from self.etd (2=pdf, 3=cdf).
            attr_name (str): The attribute name to save the result (e.g., 'pp_id2pdf_table').
        """
        lookup_table = {}

        for pp_id in np.unique(get_platform()[:, 0]):
            # Normal case
            etd_filtered = self.etd[self.etd[:, 0] == pp_id][:, [1, column_idx]]

            # Special manual handling for TaiPingYuan
            if pp_id == 103803:
                etd_filtered = self.etd[self.etd[:, 0] == 103804][:, [1, column_idx]]

            assert etd_filtered.size > 0, f"No ETD data found for pp_id {pp_id}."

            lookup_table[int(pp_id)] = etd_filtered[:, 1]

        # Set as attribute
        setattr(self, attr_name, lookup_table)

    def _build_transfer_lookup_table(self):
        """Build lookup tables for transfer: (pp_id_min, pp_id_max) -> x2cdf."""
        self.transfer_mima2cdf_table: dict[tuple[int, int], np.ndarray] = {}

        for pp_id_min, pp_id_max in (
            self.path_seg2pp_id_mima[["pp_id_min", "pp_id_max"]]
            .drop_duplicates()
            .values
        ):
            if pp_id_min == pp_id_max:  # platform swap detected, all set to one.
                self.transfer_mima2cdf_table[(int(pp_id_min), int(pp_id_max))] = (
                    np.ones(501)
                )
                continue

            ttd_filtered = self.ttd[
                (self.ttd[:, 0] == pp_id_min) & (self.ttd[:, 1] == pp_id_max)
            ][
                :, [2, 3]
            ]  # Columns: [x, cdf]

            assert (
                ttd_filtered.size > 0
            ), f"No transfer time data for pp_id_min={pp_id_min}, pp_id_max={pp_id_max}."

            self.transfer_mima2cdf_table[(int(pp_id_min), int(pp_id_max))] = (
                ttd_filtered[:, 1]
            )

    def get_egress_dis(
        self,
        path_id: int,
        times: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve egress PDF values for the given path_id and times.
        If x is out of bounds, return 0.

        Args:
            path_id (int): Path ID to retrieve the egress distribution.
            times (int | np.ndarray | pd.Series): Time(s) to retrieve PDF values.

        Returns:
            float or np.ndarray: PDF value(s) corresponding to the input time(s).
        """
        egress_pp_id = self.path_id2egress_pp_id.get(path_id, None)
        assert (
            egress_pp_id is not None
        ), f"Egress physical platform ID not found for path {path_id}."

        lookup_table = self.pp_id2pdf_table.get(egress_pp_id, 0)
        assert (
            lookup_table is not None
        ), f"No lookup table found for egress pp_id {egress_pp_id}."

        times = np.atleast_1d(times)

        max_x = lookup_table.shape[0] - 1
        valid_mask = (times >= 0) & (times <= max_x)

        output = np.zeros(times.shape)
        output[valid_mask] = lookup_table[times[valid_mask].astype(int)]

        return output if output.size > 1 else output[0]

    def get_entry_dis(
        self,
        path_id: int,
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve entry CDF values for the given path_id and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        Args:
            path_id (int): Path ID to retrieve the entry distribution.
            times_start (int | np.ndarray | pd.Series): Start time(s) to retrieve CDF values.
            times_end (int | np.ndarray | pd.Series): End time(s) to retrieve CDF values.

        Returns:
            float or np.ndarray: CDF value(s) corresponding to the input time ranges.
        """
        entry_pp_id = self.path_id2entry_pp_id.get(path_id, None)
        assert (
            entry_pp_id is not None
        ), f"Entry physical platform ID not found for path {path_id}."

        lookup_table = self.pp_id2cdf_table.get(entry_pp_id, 1)
        assert (
            lookup_table is not None
        ), f"No lookup table found for entry pp_id {entry_pp_id}."

        times_start, times_end = np.atleast_1d(times_start), np.atleast_1d(times_end)
        assert (
            times_start.shape == times_end.shape
        ), "Start and end times must have the same shape."

        cdf1, cdf2 = np.ones(times_start.shape), np.ones(times_end.shape)
        valid_mask = (times_start >= 0) & (times_end >= 0) & (times_start <= times_end)
        cdf1[valid_mask] = lookup_table[times_start[valid_mask].astype(int)]
        cdf2[valid_mask] = lookup_table[times_end[valid_mask].astype(int)]
        return cdf2 - cdf1 if cdf2.size > 1 else cdf2[0] - cdf1[0]

    def get_transfer_dis(
        self,
        path_id: int,
        seg_id: int,
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve transfer CDF values for the given path_id, seg_id, and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        Args:
            path_id (int): Path ID to retrieve the transfer distribution.
            seg_id (int): Segment ID to retrieve the transfer distribution.
            times_start (int | np.ndarray | pd.Series): Start time(s) to retrieve CDF values.
            times_end (int | np.ndarray | pd.Series): End time(s) to retrieve CDF values.

        Returns:
            float or np.ndarray: CDF value(s) corresponding to the input time ranges.
        """
        rows = (self.path_seg2pp_id_mima["path_id"] == path_id) & (
            self.path_seg2pp_id_mima["seg_id"] == seg_id
        )
        pp_ids = self.path_seg2pp_id_mima.loc[rows, ["pp_id_min", "pp_id_max"]].values
        assert (
            pp_ids.size == 2
        ), f"No or multiple pp_ids found for path {path_id} and segment {seg_id}."

        lookup_table = self.transfer_mima2cdf_table.get(
            (int(pp_ids[0][0]), int(pp_ids[0][1])), None
        )
        assert (
            lookup_table is not None
        ), f"No lookup table found for pp_id_min={pp_ids[0][0]}, pp_id_max={pp_ids[0][1]}"

        times_start, times_end = np.atleast_1d(times_start), np.atleast_1d(times_end)
        assert (
            times_start.shape == times_end.shape
        ), "Start and end times must have the same shape."

        cdf1, cdf2 = np.ones(times_start.shape), np.ones(times_end.shape)
        valid_mask = (times_start >= 0) & (times_end >= 0) & (times_start <= times_end)
        cdf1[valid_mask] = lookup_table[times_start[valid_mask].astype(int)]
        cdf2[valid_mask] = lookup_table[times_end[valid_mask].astype(int)]
        return cdf2 - cdf1 if cdf2.size > 1 else cdf2[0] - cdf1[0]


def test_one_feas_iti():
    config.load_config()
    from src.utils import read_
    from src.globals import get_afc

    wtdc = WalkTimeDisCalculator()

    # randomly select one feasible itinerary
    df_left = read_("left", latest_=False, show_timer=False)
    rid = np.random.choice(df_left["rid"].values, size=1)[0]
    iti_id = np.random.choice(df_left[df_left["rid"] == rid]["iti_id"].values, size=1)[
        0
    ]
    rid, iti_id = 950850, 26  # manual set
    print(f"rid: {rid}. iti_id: {iti_id}.")

    df = df_left[(df_left["rid"] == rid) & (df_left["iti_id"] == iti_id)]
    print(df)
    path_id = df["path_id"].values[0]

    rid, uid1, ts1, uid2, ts2 = get_afc()[get_afc()[:, 0] == rid][0]
    print(rid, uid1, ts1, uid2, ts2)

    entry_dis = wtdc.get_entry_dis(path_id, 0, df.iloc[0, 5] - ts1)
    egress_dis = wtdc.get_egress_dis(path_id, ts2 - df.iloc[-1, 6])
    # transfer_dis = wtdc.get_transfer_dis(path_id, seg_id, 0, df.iloc[1:-1, 5] - ts1)
    for i in range(len(df) - 1):
        if i == 0:
            prev_ts = ts1
        else:
            prev_ts = df.iloc[i - 1, 6]
        print(path_id, df.iloc[i, 3], prev_ts, df.iloc[i, 5], end=" | ")
        transfer_dis = wtdc.get_transfer_dis(
            path_id, df.iloc[i, 3], 0, df.iloc[i, 5] - prev_ts
        )
        print(transfer_dis)

    print(entry_dis, egress_dis)


if __name__ == "__main__":
    config.load_config()
    pass
