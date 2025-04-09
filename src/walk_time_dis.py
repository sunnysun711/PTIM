"""
This script analyzes walking time distributions at each station UID
based on assigned and candidate feasible itineraries.

Data sources:
- 'feas_iti_assigned.pkl': assigned itineraries (only one per passenger).
- 'feas_iti_left.pkl': itineraries sharing the same alighting train but not selected.

Main steps:
1. Combine walking time samples from both files by station UID.
2. Compute the probability density function (PDF) and cumulative distribution function (CDF).
3. Save the statistical results to a CSV file for further analysis or visualization.

Usage:
Import and call the following functions as needed:
- `get_pdf()`: Calculate PDF from walking time samples.
- `get_cdf()`: Calculate CDF from walking time samples.
"""
import numpy as np

from src.utils import read_data, read_data_all


def find_rids_whose_feas_iti_all_share_same_alighting_trains_in_feas_iti_left_pkl():
    ...



def fit_walk_time_distribution():

    ...


def get_pdf(walk_param: dict[str, float], walk_time: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the probability density function (PDF) of walking time.

    Parameters
    ----------
    walk_param : dict[str, float]
        Dictionary containing parameters of the walking time distribution.
        Must include keys: 'mu', 'sigma' (for log-normal distribution).

    walk_time : float or Iterable[float]
        Actual walking time(s) to evaluate the PDF. Can be a scalar or a 1D array-like object.

    Returns
    -------
    float or np.ndarray
        PDF value(s) corresponding to the input walking time(s).

    Notes
    -----
    This function is vectorized for performance and can operate efficiently over entire columns.
    """
    ...


def get_cdf(walk_param: dict[str, float], t_start: float | np.ndarray, t_end: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the cumulative distribution function (CDF) of walking time between t_start and t_end.

    Parameters
    ----------
    walk_param : dict[str, float]
        Dictionary containing parameters of the walking time distribution.
        Must include keys: 'mu', 'sigma' (for log-normal distribution).

    t_start : float or Iterable[float]
        Start time(s) of the interval. Can be a scalar or 1D array-like object.

    t_end : float or Iterable[float]
        End time(s) of the interval. Must be the same shape as `t_start`.

    Returns
    -------
    float or np.ndarray
        CDF value(s) for the interval [t_start, t_end].

    Notes
    -----
    Computes P(walk_time ∈ [t_start, t_end]).
    Efficiently supports vectorized input for use with entire Series/arrays.
    """
    ...
