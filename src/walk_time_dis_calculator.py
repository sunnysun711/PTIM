"""
walk_time_dis_calculator.py

A module for efficient and scalable computation of walking time distributions
in metro systems, covering entry, egress, and transfer scenarios.

This module defines the `WalkTimeDisModel` class, which provides methods to
retrieve probability distribution values (PDF/CDF) for walking times based
on a passenger's travel path and segment. All distribution data is preloaded
into lookup tables for fast, vectorized access.

Key Features:
-------------
- Computes:
    - Egress time probability density function (PDF)
    - Entry time cumulative distribution function (CDF)
    - Transfer time cumulative distribution function (CDF)
- Uses NumPy arrays and precomputed lookup tables for high-performance querying
- Handles platform-specific exceptions (e.g., special case for TaiPingYuan)
- Provides access both by physical platform ID and path/segment IDs

Main Class:
-----------
- WalkTimeDisModel

    Key Methods:
    - compute_egress_pdf(path_id, times, ratio_=True, square_=False): 
        Retrieve egress PDF for a path at given times
    - compute_entry_cdf(path_id, t_start, t_end): 
        Retrieve entry CDF for a path within a time range
    - compute_transfer_cdf(path_id, seg_id, t_start, t_end): 
        Retrieve transfer CDF for a path-segment within a time range
    - compute_egress_pdf_from_pp(pp_id, times, ratio_, square_): 
        Retrieve egress PDF by platform ID
    - compute_entry_cdf_from_pp(pp_id, t_start, t_end): 
        Retrieve entry CDF by platform ID
    - compute_transfer_cdf_from_pp(pp_id_min, pp_id_max, t_start, t_end): 
        Retrieve transfer CDF by platform pair

Testing Utilities:
------------------
- _test_feas_iti_dis_calculate(rid, iti_id, df_left): 
    Print and return probability components for a specific itinerary
- _test_feas_iti_dis_calculate_one_rid(rid, plot_seg_trains=True): 
    Print and summarize all itineraries for a record ID (optional plotting)

Data Requirements:
------------------
- ETD (Egress Time Distribution): ndarray, shape (N, 4)
    [pp_id, x, pdf, cdf]
- TTD (Transfer Time Distribution): ndarray, shape (M, 4)
    [pp_id_min, pp_id_max, x, cdf]
- Platform mapping: from get_platform()
- K-shortest path info: from get_k_pv()
- Path-segment platform mapping: from get_path_seg_to_pp_ids()

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
- src.utils
- src.walk_time_filter

Usage Example:
--------------
>>> from walk_time_dis_model import WalkTimeDisModel
>>> model = WalkTimeDisModel()
>>> pdf = model.compute_egress_pdf(path_id=1001, times=[10, 20, 30])
>>> cdf = model.compute_entry_cdf(path_id=1001, t_start=5, t_end=25)
>>> cdf_transfer = model.compute_transfer_cdf(path_id=1001, seg_id=2, t_start=10, t_end=35)
"""

import numpy as np
import pandas as pd

from src import config
from src.globals import get_etd, get_ttd, get_k_pv, get_platform
from src.utils import read_all
from src.walk_time_filter import get_path_seg_to_pp_ids


class WalkTimeDisModel:
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

        :param etd: (optional) Egress Time Distribution (ETD) data.
        :type etd: np.ndarray, optional

        :param ttd: (optional) Transfer Time Distribution (TTD) data.
        :type ttd: np.ndarray, optional
        """
        # Load ETD and TTD
        self.etd: np.ndarray = etd if etd is not None else get_etd()
        self.ttd: np.ndarray = ttd if ttd is not None else get_ttd()

        # Initialize mappings
        self._build_path_entry_egress_map()
        self._build_transfer_map()

        # Build lookup tables
        self._create_pp_id_lookup(2, "pp_id2pdf_table")  # Egress PDF
        self._create_pp_id_lookup(3, "pp_id2cdf_table")  # Entry CDF
        self._create_transfer_lookup()  # Transfer CDF

    def _build_path_entry_egress_map(self):
        """Builds mapping from path_id to entry/egress physical platform ids."""
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

    def _build_transfer_map(self):
        """Builds mapping from (path_id, seg_id) to transfer pp_id_min and pp_id_max."""
        df = get_path_seg_to_pp_ids()
        df["pp_id_min"] = df[["pp_id1", "pp_id2"]].min(axis=1)
        df["pp_id_max"] = df[["pp_id1", "pp_id2"]].max(axis=1)
        self.path_seg2pp_id_mima: pd.DataFrame = df.drop(
            columns=["pp_id1", "pp_id2"])

    def _create_pp_id_lookup(
        self,
        column_idx: int,
        attr_name: str,
    ):
        """
        General function to build lookup tables for pp_id -> distribution values.

        :param column_idx: The column index to extract from self.etd (2=pdf, 3=cdf).
        :type column_idx: int
        :param attr_name: The attribute name to save the result (e.g., 'pp_id2pdf_table').
        :type attr_name: str
        """
        lookup_table = {}

        for pp_id in np.unique(get_platform()[:, 0]):
            # Normal case
            etd_filtered = self.etd[self.etd[:, 0]
                                    == pp_id][:, [1, column_idx]]

            # Special manual handling for TaiPingYuan
            if pp_id == 103803:
                etd_filtered = self.etd[self.etd[:, 0]
                                        == 103804][:, [1, column_idx]]

            assert etd_filtered.size > 0, f"No ETD data found for pp_id {pp_id}."

            lookup_table[int(pp_id)] = etd_filtered[:, 1]

        # Set as attribute
        setattr(self, attr_name, lookup_table)

    def _create_transfer_lookup(self):
        """Build lookup tables for transfer: (pp_id_min, pp_id_max) -> x2cdf."""
        self.transfer_mima2cdf_table: dict[tuple[int, int], np.ndarray] = {}

        for pp_id_min, pp_id_max in (
            self.path_seg2pp_id_mima[["pp_id_min", "pp_id_max"]]
            .drop_duplicates()
            .values
        ):
            if pp_id_min == pp_id_max:  # platform swap, all set to one (except location zero).
                _ = np.ones(501)
                _[0] = 0
                self.transfer_mima2cdf_table[(int(pp_id_min), int(pp_id_max))] = _
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

    def compute_egress_pdf_from_pp(
        self,
        pp_id: int,
        times: int | np.ndarray | pd.Series, 
        ratio_: bool = True,
        square_: bool = False,
    ) -> float | np.ndarray:
        """
        Retrieve egress PDF values for the given pp_id and times.
        If x is out of bounds, return 0.

        :param pp_id: Physical platform ID to retrieve the egress distribution.
        :type pp_id: int

        :param times: Time(s) to retrieve PDF values.
        :type times: int | np.ndarray | pd.Series

        :param ratio_: Whether to return the ratio of the PDF to the max PDF value.
        :type ratio_: bool, optional, default=True

        :param square_: Whether to square the output values.
        :type square_: bool, optional, default=False

        :returns: PDF value(s) corresponding to the input time(s).
        :rtype: float | np.ndarray

        :raises Error: If no lookup table found for the given egress pp_id.
        """
        lookup_table = self.pp_id2pdf_table.get(pp_id, None)
        assert (
            lookup_table is not None
        ), f"No lookup table found for egress pp_id {pp_id}."

        times = np.atleast_1d(times)

        max_x = lookup_table.shape[0] - 1
        valid_mask = (times >= 0) & (times <= max_x)

        output = np.zeros(times.shape)
        output[valid_mask] = lookup_table[times[valid_mask].astype(int)]
        if ratio_:
            output /= np.max(lookup_table)
        if square_:
            output **= 2

        return output if output.size > 1 else output[0]

    def compute_egress_pdf(
        self,
        path_id: int,
        times: int | np.ndarray | pd.Series,
        ratio_: bool = True,
        square_: bool = False,
    ) -> float | np.ndarray:
        """
        Retrieve egress PDF values for the given path_id and times.
        If x is out of bounds, return 0.

        :param path_id: Path ID to retrieve the egress distribution.
        :type path_id: int

        :param times: Time(s) to retrieve PDF values.
        :type times: int | np.ndarray | pd.Series

        :param ratio_: Whether to return the ratio to the max PDF value.
        :type ratio_: bool, optional, default=True

        :param square_: Whether to square the output values.
        :type square_: bool, optional, default=False

        :returns: PDF value(s) corresponding to the input time(s).
        :rtype: float | np.ndarray
        """
        egress_pp_id = self.path_id2egress_pp_id.get(path_id, None)
        assert (
            egress_pp_id is not None
        ), f"Egress physical platform ID not found for path {path_id}."
        self.compute_egress_pdf_from_pp(egress_pp_id, times, ratio_, square_)
        

    def _lookup_time_range_deltas(
        self,
        lookup_table: np.ndarray,
        t_start: int | np.ndarray | pd.Series,
        t_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Look up time range deltas in the given lookup table.

        :param lookup_table: Lookup table containing CDF values.
        :type lookup_table: np.ndarray

        :param t_start: Start time(s) for lookup.
        :type t_start: int | np.ndarray | pd.Series

        :param t_end: End time(s) for lookup.
        :type t_end: int | np.ndarray | pd.Series

        :returns: Time range delta(s) corresponding to the input time ranges.
        :rtype: float | np.ndarray
        """
        t_start, t_end = np.atleast_1d(t_start), np.atleast_1d(t_end)
        assert (
            t_start.shape == t_end.shape
        ), "Start and end times must have the same shape."
        assert np.all(t_start >= 0) or np.all(
            t_end >= 0), "Negative times detected."

        max_x = lookup_table.shape[0] - 1
        t_start = np.clip(t_start, 0, max_x) if np.any(
            t_start > max_x) else t_start
        t_end = np.clip(t_end, 0, max_x) if np.any(t_end > max_x) else t_end

        cdf1, cdf2 = np.zeros(t_start.shape), np.ones(t_end.shape)
        valid_mask = t_start <= t_end

        cdf1[valid_mask] = lookup_table[t_start[valid_mask].astype(int)]
        cdf2[valid_mask] = lookup_table[t_end[valid_mask].astype(int)]

        return cdf2 - cdf1 if cdf2.size > 1 else cdf2[0] - cdf1[0]

    def compute_entry_cdf_from_pp(
        self,
        pp_id: int,
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve entry CDF values for the given pp_id and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        :param pp_id: Physical platform ID to retrieve the entry distribution.
        :type pp_id: int

        :param times_start: Start time(s) to retrieve CDF values.
        :type times_start: int | np.ndarray | pd.Series

        :param times_end: End time(s) to retrieve CDF values.
        :type times_end: int | np.ndarray | pd.Series

        :returns: CDF value(s) corresponding to the input time ranges.
        :rtype: float | np.ndarray
        """
        lookup_table = self.pp_id2cdf_table.get(pp_id, None)
        assert (
            lookup_table is not None
        ), f"No lookup table found for entry pp_id {pp_id}."

        return self._lookup_time_range_deltas(lookup_table, times_start, times_end)
        ...

    def compute_entry_cdf(
        self,
        path_id: int,
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve entry CDF values for the given path_id and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        :param path_id: Path ID to retrieve the entry distribution.
        :type path_id: int

        :param times_start: Start time(s) to retrieve CDF values.
        :type times_start: int | np.ndarray | pd.Series

        :param times_end: End time(s) to retrieve CDF values.
        :type times_end: int | np.ndarray | pd.Series

        :returns: CDF value(s) corresponding to the input time ranges.
        :rtype: float | np.ndarray
        """
        entry_pp_id = self.path_id2entry_pp_id.get(path_id, None)
        assert (
            entry_pp_id is not None
        ), f"Entry physical platform ID not found for path {path_id}."
        return self.compute_entry_cdf_from_pp(entry_pp_id, times_start, times_end)
        

    def compute_transfer_cdf_from_pp(
        self,
        pp_id_min: int,
        pp_id_max: int, 
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve transfer CDF values for the given pp_id_min, pp_id_max, and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        :param pp_id_min: The smaller physical platform ID to retrieve the transfer distribution.
        :type pp_id_min: int

        :param pp_id_max: The larger physical platform ID to retrieve the transfer distribution.
        :type pp_id_max: int

        :param times_start: Start time(s) to retrieve CDF values.
        :type times_start: int | np.ndarray | pd.Series

        :param times_end: End time(s) to retrieve CDF values.
        :type times_end: int | np.ndarray | pd.Series

        :returns: CDF value(s) corresponding to the input time ranges.
        :rtype: float | np.ndarray
        """
        lookup_table = self.transfer_mima2cdf_table.get(
            (pp_id_min, pp_id_max), None
        )
        assert (
            lookup_table is not None
        ), f"No lookup table found for pp_id_min={pp_id_min}, pp_id_max={pp_id_max}"

        return self._lookup_time_range_deltas(lookup_table, times_start, times_end)
        ...

    def compute_transfer_cdf(
        self,
        path_id: int,
        seg_id: int,
        times_start: int | np.ndarray | pd.Series,
        times_end: int | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """
        Retrieve transfer CDF values for the given path_id, seg_id, and time ranges.
        If times_start or times_end is out of bounds, return 1.0 as CDF values.

        :param path_id: Path ID to retrieve the transfer distribution.
        :type path_id: int

        :param seg_id: Segment ID to retrieve the transfer distribution.
        :type seg_id: int

        :param times_start: Start time(s) to retrieve CDF values.
        :type times_start: int | np.ndarray | pd.Series

        :param times_end: End time(s) to retrieve CDF values.
        :type times_end: int | np.ndarray | pd.Series

        :returns: CDF value(s) corresponding to the input time ranges.
        :rtype: float | np.ndarray
        """
        rows = (self.path_seg2pp_id_mima["path_id"] == path_id) & (
            self.path_seg2pp_id_mima["seg_id"] == seg_id
        )
        pp_ids = self.path_seg2pp_id_mima.loc[rows, [
            "pp_id_min", "pp_id_max"]].values
        assert (
            pp_ids.size == 2
        ), f"No or multiple pp_ids found for path {path_id} and segment {seg_id}."

        return self.compute_transfer_cdf_from_pp(
            int(pp_ids[0][0]), int(pp_ids[0][1]), times_start, times_end
        )


def _test_feas_iti_dis_calculate(rid: int, iti_id: int, df_left: pd.DataFrame) -> list[float]:
    """
    Compute and print probability components for a given (rid, iti_id) feasible itinerary.

    :param rid: Passenger record ID.
    :type rid: int

    :param iti_id: Itinerary ID.
    :type iti_id: int

    :param df_left: DataFrame of all feasible itineraries (filtered from 'left').
    :type df_left: pd.DataFrame

    :returns: List containing the entry, egress, transfer product, and egress square probabilities.
    :rtype: list[float]
    """
    from src.utils import ts2tstr
    from src.globals import get_afc

    afc = get_afc()
    def to_str(x): return ts2tstr(x, include_seconds=True)
    wtdc = WalkTimeDisModel()

    df = df_left[(df_left["rid"] == rid) & (df_left["iti_id"] == iti_id)]
    if df.empty:
        print(f"[Warning] No itinerary found for rid={rid}, iti_id={iti_id}")
        return [0, 0, 0, 0]

    path_id = df["path_id"].values[0]
    rid, uid1, ts1, uid2, ts2 = afc[afc[:, 0] == rid][0]

    print(f"\n=== RID: {rid}, ITI_ID: {iti_id} ===, Path ID: {path_id} ===")

    # Entry probability
    entry_dis = wtdc.compute_entry_cdf(path_id, 0, df.iloc[0, 5] - ts1)
    print(f"\tEntry  : {to_str(ts1)} -> {to_str(df.iloc[0, 5])} "
          f"({df.iloc[0, 5] - ts1:4}) | {entry_dis:.6f}")

    # Transfer probabilities
    transfer_dis_product = 1.0
    for i in range(len(df) - 1):
        transfer_dis = wtdc.compute_transfer_cdf(
            path_id, df.iloc[i, 3], 0, df.iloc[i + 1, 5] - df.iloc[i, 6]
        )
        print(f"\tTrans {df.iloc[i, 3]}: {to_str(df.iloc[i, 6])} -> {to_str(df.iloc[i + 1, 5])} "
              f"({df.iloc[i + 1, 5] - df.iloc[i, 6]:4}) | {transfer_dis:.6f}")
        transfer_dis_product *= transfer_dis

    # Egress probabilities
    egress_dis = wtdc.compute_egress_pdf(
        path_id, ts2 - df.iloc[-1, 6], ratio_=True, square_=False)
    egress_dis_square = wtdc.compute_egress_pdf(
        path_id, ts2 - df.iloc[-1, 6], square_=True, ratio_=True)
    print(f"\tEgress : {to_str(df.iloc[-1, 6])} -> {to_str(ts2)} "
          f"({ts2 - df.iloc[-1, 6]:4}) | {egress_dis:.6f} | {egress_dis_square:.6f}")

    return [entry_dis, egress_dis, transfer_dis_product, egress_dis_square]


def _test_feas_iti_dis_calculate_one_rid(rid: int, plot_seg_trains: bool = True):
    """
    Evaluate and summarize all feasible itineraries for a given RID.
    Plot all segment-level trains after probability printing if enabled.

    :param rid: Passenger record ID.
    :type rid: int

    :param plot_seg_trains: Whether to plot segment-level trains for visual inspection.
    :type plot_seg_trains: bool, optional, default=True
    """
    from src.utils import read_
    from src.passenger import _plot_check_feas_iti

    df_left = read_("left", latest_=False, show_timer=False)

    if rid is None:
        rid = np.random.choice(df_left["rid"].unique())
    num_iti = df_left[df_left["rid"] == rid]["iti_id"].nunique()
    print(f"RID, ITI_NUM: {rid}, {num_iti}")
    df_left = df_left[df_left["rid"] == rid]

    results = []
    for iti_id in range(1, num_iti+1):
        dis_list = _test_feas_iti_dis_calculate(rid, iti_id, df_left)
        results.append(dis_list)

    results = np.array(results)

    # Print nicely
    np.set_printoptions(
        precision=6,
        suppress=True,
        linewidth=180,
        threshold=np.inf,
        floatmode='maxprec_equal'
    )
    pd.set_option('display.precision', 6)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame(results, columns=[
                      "entry_cdf", "egress_pdf", "trans_cdf_prod", "egress_pdf2"])
    df["prod"] = df["entry_cdf"] * df["trans_cdf_prod"] * df["egress_pdf"]
    df["prob"] = df["prod"] / df["prod"].sum()
    df["prod2"] = df["entry_cdf"] * df["trans_cdf_prod"] * df["egress_pdf2"]
    df["prob2"] = df["prod2"] / df["prod2"].sum()
    # print(df)
    print(df[df["prob"] > 1e-3])

    if plot_seg_trains:
        _plot_check_feas_iti(rid=rid, print_on=False)


if __name__ == "__main__":
    config.load_config()
    _test_feas_iti_dis_calculate_one_rid(rid=None)  # 1723090

    pass
