"""
This module contains functions for fitting statistical distributions to walking time data,
evaluating the goodness of fit, and applying the fitted models to make probabilistic
calculations about walking times.

Functions:
- `fit_pdf_cdf()`: Fit a probability density function (PDF) and cumulative distribution
  function (CDF) to the input data using methods like Kernel Density Estimation (KDE),
  Gamma distribution, or Log-Normal distribution.
- `evaluate_fit()`: Evaluate the fit of a cumulative distribution function (CDF) to the input data
  using the Kolmogorov-Smirnov (K-S) test.
- `fit_one_pl()`: Fit the distribution of egress time for a specific physical link.
- `fit_egress_time_dis_all_parallel()`: Fit the distribution of egress time for all physical links in parallel.
- `fit_transfer_time_dis_all()`: Fit the distribution of transfer time for all transfer links.
"""

from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import kstest
from joblib import Parallel, delayed
from tqdm import tqdm


def reject_outlier_bd(data: np.ndarray, method: str = "zscore", abs_max: int | None = 500) -> tuple[float, float]:
    """
    Calculate bounds for outlier rejection.
    see:
        boxplot: https://www.secrss.com/articles/11994
        zscore: https://www.zhihu.com/question/38066650

    :param data: Input data array.
    :param method: Outlier detection method ('zscore' or 'boxplot').
    :param abs_max: Absolute maximum value constraint.
    :return: A tuple of lower and upper bounds for valid data.

    Raises:
        Exception: If an invalid method is provided.
    """
    if method == "boxplot":
        miu = np.mean(data)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        lower_whisker = Q1 - (miu - Q1)  # lower limit
        upper_whisker = Q3 + (Q3 - miu)  # upper limit
        lower_bound, upper_bound = lower_whisker, upper_whisker
    elif method == "zscore":
        lower_bound, upper_bound = np.mean(
            data) - 3 * np.std(data), np.mean(data) + 3 * np.std(data)
    else:
        raise Exception(
            "Please use either boxplot or zscore method to reject outliers!")

    # manual bound
    if abs_max:
        upper_bound = min(upper_bound, abs_max)
    lower_bound = max(lower_bound, 0)

    return lower_bound, upper_bound


def reject_outlier(data: np.ndarray, method: str = "zscore", abs_max: int = 500) -> np.ndarray:
    """
    Reject outliers from the input data array based on the specified method.
    see:
        boxplot: https://www.secrss.com/articles/11994
        zscore: https://www.zhihu.com/question/38066650

    :param data: Input data array.
    :param method: Outlier detection method ('zscore' or 'boxplot').
    :param abs_max: Absolute maximum value constraint.
    :return: cleaned data array.

    Raises:
        Exception: If an invalid method is provided.
    """
    lb, ub = reject_outlier_bd(data, method=method, abs_max=abs_max)
    return data[(data >= lb) & (data <= ub)]


def fit_pdf_cdf(data: np.ndarray, method: str = "kde") -> tuple[Callable, Callable]:
    """
    Fit a probability density function (PDF) and cumulative distribution function (CDF) to the input data.

    Parameters:
        data (np.ndarray): Input data array.
        method (str): Method to fit the PDF and CDF. Options are 'kde' (Kernel Density Estimation),
                      'gamma' (Gamma Distribution), and 'lognorm' (Log-Normal Distribution).

    Returns:
        tuple[Callable, Callable]: A tuple containing the fitted PDF function and CDF function.
                                   The PDF function takes x values as input and returns the corresponding
                                   PDF values. The CDF function takes x values as input and returns the
                                   corresponding CDF values.

    Raises:
        Exception: If an invalid method is provided.
    """
    if method == "kde":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        return kde, lambda x_values: np.array([kde.integrate_box_1d(0, x) for x in x_values])
    elif method == "gamma":
        from scipy.stats import gamma
        params = gamma.fit(data, floc=0)
        # print(params)
        return gamma(*params).pdf, gamma(*params).cdf
    elif method == "lognorm":
        from scipy.stats import lognorm
        params = lognorm.fit(data)
        # print(params)
        return lognorm(*params).pdf, lognorm(*params).cdf
    else:
        raise Exception(
            "Please use either kde, gamma, or lognorm method to fit pdf!")


def evaluate_fit(data: np.ndarray, cdf_func: Callable) -> tuple[float, float]:
    """
    Evaluate the fit of a cumulative distribution function (CDF) to the input data.

    :param data: Input data array.
    :param cdf_func: CDF function to evaluate.
    :return: A tuple containing the K-S statistic and the p-value of the K-S test.
    """
    data_sorted = np.sort(data)
    return kstest(data_sorted, cdf_func)


def fit_one_pl(pl_id: int, et_: pd.DataFrame, physical_link_info: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """
    Fit the distribution of physical links egress time with the following columns:
        [
            pl_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]
    :param pl_id: physical link id
    :param et_: egress time dataframe
    :param physical_link_info: physical link info array, each row is [pl_id, platform_id, uid]
    :param x: x values for pdf and cdf, usually [0, 500] with 501 points

    :return: array with shape (x.size, 14) or None
    """
    et = et_[et_["node1"].isin(
        physical_link_info[physical_link_info[:, 0] == pl_id][:, 1])]
    if et.shape[0] == 0:
        return None
    data = et['egress_time'].values
    data = reject_outlier(data, method="zscore", abs_max=500)
    res_this_pl = [np.ones_like(x) * pl_id, x]

    for met in ["kde", "gamma", "lognorm"]:
        pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
        pdf_values = pdf_f(x)
        cdf_values = cdf_f(x)
        cdf_values = cdf_values / cdf_values[-1]  # normalize
        ks_stat, ks_p_val = evaluate_fit(data=data, cdf_func=cdf_f)
        res_this_pl.extend([pdf_values, cdf_values,
                            np.ones_like(x) * ks_stat,
                            np.ones_like(x) * ks_p_val])

    return np.vstack(res_this_pl).T


def fit_egress_time_dis_all_parallel(
        et_: pd.DataFrame,
        physical_link_info: np.ndarray = None,
        n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fit the distribution of physical links egress time with the following columns:
        [
            pl_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]
    :param et_: egress time dataframe
    :param physical_link_info: physical link info array, each row is [pl_id, platform_id, uid]
    :param n_jobs: number of jobs to run in parallel, default is -1 (use all available cores)

    :return: dataframe with shape (n_pl * 501, 14)
    """
    print(
        f"[INFO] Start fitting egress time distribution using {n_jobs} threads...")

    x = np.linspace(0, 500, 501)
    physical_link_info = physical_link_info if physical_link_info is not None else get_physical_links_info(
        et_=et_)
    pl_ids = np.unique(physical_link_info[:, 0])

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_one_pl)(pl_id, et_, physical_link_info, x) for pl_id in
        tqdm(pl_ids, desc="Physical links egress time distribution fitting")
    )
    results = [res for res in results if res is not None]
    res = np.vstack(results)

    return pd.DataFrame(res, columns=[
        "pl_id", "x",
        "kde_pdf", "kde_cdf", "kde_ks_stat", "kde_ks_p_value",
        "gamma_pdf", "gamma_cdf", "gamma_ks_stat", "gamma_ks_p_value",
        "lognorm_pdf", "lognorm_cdf", "lognorm_ks_stat", "lognorm_ks_p_value"
    ])


def fit_transfer_time_dis_all(df_tt: pd.DataFrame, df_ps2t: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the distribution of transfer time for all transfer links.

    :param df_tt: Filtered transfer time DataFrame with 4 columns:
        ["path_id", "seg_id", "alight_ts", "transfer_time"]
        where seg_id is the alighting train segment id, transfer_time is the time difference between the alighting
        time of seg_id and the boarding time of seg_id + 1.
    :param df_ps2t: Generated transfer info mapping from path seg, DataFrame with 7 columns:
        ["path_id", "seg_id", "node1", "node2", "p_uid_min", "p_uid_max", "transfer_type"]
        where seg_id is the alighting train segment id, transfer_type is one of "platform_swap", "egress-entry",
        and p_uid_min, p_uid_max are the platform_uids of the two platforms involved in the transfer.

    :return: DataFrame with 6 columns:
        [p_uid1, p_uid2, x, kde_cdf, gamma_cdf, lognorm_cdf]
        where p_uid1, p_uid2 are smaller and larger platform_uid of the transfer link.
    """
    print("[INFO] Start fitting transfer time distribution...")
    # merge df_tt and df_p2t to get (p_uid_min, p_uid_max) -> transfer_time info
    df = pd.merge(df_tt, df_ps2t, on=["path_id", "seg_id"], how="left")

    x = np.linspace(0, 500, 501)
    res = []
    # for egress-entry transfers:
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "egress-entry"].groupby(["p_uid_min", "p_uid_max"])[
            "transfer_time"]:
        data = df_.values
        print(p_uid1, p_uid2, data.size, end=" -> ")
        data = reject_outlier(data, abs_max=500)
        print(data.size, end="\t | ")

        res_this_transfer = [np.ones_like(
            x) * p_uid1, np.ones_like(x) * p_uid2, x]
        for met in ["kde", "gamma", "lognorm"]:
            pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
            cdf_values = cdf_f(x)
            cdf_values = cdf_values / cdf_values[-1]  # normalize
            ks_stat, ks_p_val = evaluate_fit(data=data, cdf_func=cdf_f)
            print(f"{met}: {ks_stat:.4f} {ks_p_val:.4f}", end=" | ")
            res_this_transfer.extend([cdf_values])
        res.append(np.vstack(res_this_transfer).T)
        print()
    # for platform_swap transfers: just defaults to 1
    for (p_uid1, p_uid2), df_ in df[df["transfer_type"] == "platform_swap"].groupby(["p_uid_min", "p_uid_max"])[
            "transfer_time"]:
        data = df_.values
        print(p_uid1, p_uid2, data.size)
        res.append(np.array([[
            p_uid1, p_uid2, 0, 1, 1, 1
        ]]))
    res = np.vstack(res)
    res = pd.DataFrame(res, columns=[
        "p_uid1", "p_uid2", "x",
        "kde_cdf", "gamma_cdf", "lognorm_cdf"
    ])
    for col in ["p_uid1", "p_uid2", "x"]:
        res[col] = res[col].astype(int)
    return res
