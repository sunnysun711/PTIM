from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import kstest
from joblib import Parallel, delayed
from tqdm import tqdm

from src.globals import get_platform


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


def fit_one_physical_platform(pp_id: int, eg_t_data: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """
    Fit the distribution of physical platform egress time with the following columns:
        [
            pp_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]

    Parameters:
    --------
    :param pp_id: Physical platform ID.
    :param eg_t_data: Egress time numpy array of the current physical platform.
    :param x: X values for PDF and CDF, usually [0, 500] with 501 points.

    Returns:
    --------
    :return: np.ndarray | None: Array with shape (x.size, 14) or None.
    """
    if eg_t_data.size == 0:
        return None
    data = reject_outlier(eg_t_data, method="zscore", abs_max=500)
    res_this_pp = [np.ones_like(x) * pp_id, x]

    for met in ["kde", "gamma", "lognorm"]:
        pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
        pdf_values = pdf_f(x)
        cdf_values = cdf_f(x)
        cdf_values = cdf_values / cdf_values[-1]  # normalize
        ks_stat, ks_p_val = evaluate_fit(data=data, cdf_func=cdf_f)
        res_this_pp.extend([pdf_values, cdf_values,
                            np.ones_like(x) * ks_stat,
                            np.ones_like(x) * ks_p_val])
    return np.vstack(res_this_pp).T


def fit_platform_egress_time_dis_all_parallel(
        eg_t: pd.DataFrame, n_jobs: int = -1
):
    """
    Fit the distribution of physical platform egress time with the following columns:
        [
            pp_id, x,
            kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
            gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
            lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value
        ]

    :param eg_t: Egress time dataframe.
        Each row is [rid (index), physical_platform_id, alight_ts, egress_time].
    :param n_jobs: Number of threads to use. Default is -1, which uses all available threads.

    :return: pd.DataFrame: Dataframe with shape (x.size * pp_id.size, 14).
        Each row is [pp_id, x, kde_pdf, kde_cdf, kde_ks_stat, kde_ks_p_value,
                    gamma_pdf, gamma_cdf, gamma_ks_stat, gamma_ks_p_value,
                    lognorm_pdf, lognorm_cdf, lognorm_ks_stat, lognorm_ks_p_value].
    """
    print(
        f"[INFO] Start fitting egress time distribution using {n_jobs} threads...")

    x = np.linspace(0, 500, 501)
    platform: np.ndarray = get_platform()  # [pp_id, node_id, uid]

    pp_ids = np.unique(platform[:, 0])

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_one_physical_platform)(
            pp_id=pp_id,
            eg_t_data=eg_t[eg_t["physical_platform_id"] == pp_id]["egress_time"].values,
            x=x,
        )
        for pp_id in tqdm(pp_ids, desc="Egress time distribution fitting for each physical platform")
    )
    results = [res for res in results if res is not None]
    res = np.vstack(results)
    return pd.DataFrame(res, columns=[
        "pp_id", "x",
        "kde_pdf", "kde_cdf", "kde_ks_stat", "kde_ks_p_value",
        "gamma_pdf", "gamma_cdf", "gamma_ks_stat", "gamma_ks_p_value",
        "lognorm_pdf", "lognorm_cdf", "lognorm_ks_stat", "lognorm_ks_p_value"
    ])


def fit_transfer_time_dis_all(tr_t: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the distribution of transfer time for all transfer links.
    :param tr_t: Transfer time dataframe.
        Columns: [rid(index), path_id, seg_id, pp_id1, pp_id2, alight_ts, transfer_time, transfer_type]
    :return: Dataframe with columns:
        [pp_id_min, pp_id_max, x, kde_cdf, gamma_cdf, lognorm_cdf]
        where pp_id_min and pp_id_max are the physical platform ids of the transfer link,
        x is the transfer time, and kde_cdf, gamma_cdf, lognorm_cdf are the CDF values of the transfer time distribution
        fitted with KDE, Gamma, and Log-Normal distributions respectively.
        Each row is a transfer link.
        Note: for platform_swap transfers (pp_id_min=pp_id_max), the CDF values are all 1, and x is only one value as 0.
        Note: for egress-entry transfers, the CDF values are normalized to [0, 1].
    """
    print("[INFO] Start fitting transfer time distribution...")
    platform: np.ndarray = get_platform()  # [pp_id, node_id, uid]

    tr_t["pp_id_min"] = tr_t[["pp_id1", "pp_id2"]].min(axis=1)
    tr_t["pp_id_max"] = tr_t[["pp_id1", "pp_id2"]].max(axis=1)

    x = np.linspace(0, 500, 501)
    res = []
    # for egress-entry transfers:
    for (pp_id_min, pp_id_max), df_ in tr_t[tr_t["transfer_type"] == "egress-entry"].groupby(
            ["pp_id_min", "pp_id_max"])["transfer_time"]:
        data = df_.values
        print(pp_id_min, pp_id_max, data.size, end=" -> ")
        data = reject_outlier(data, abs_max=500)
        print(data.size, end="\t | ")

        res_this_transfer = [
            np.ones_like(x) * pp_id_min,
            np.ones_like(x) * pp_id_max,
            x
        ]
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
    for (pp_id_min, pp_id_max), df_ in tr_t[tr_t["transfer_type"] == "platform_swap"].groupby(
            ["pp_id_min", "pp_id_max"])["transfer_time"]:
        data = df_.values
        print(pp_id_min, pp_id_max, data.size)
        res.append(np.array([[
            pp_id_min, pp_id_max, 0, 1, 1, 1
        ]]))
    res = np.vstack(res)
    res = pd.DataFrame(res, columns=[
        "pp_id_min", "pp_id_max", "x",
        "kde_cdf", "gamma_cdf", "lognorm_cdf"
    ])
    for col in ["pp_id_min", "pp_id_max", "x"]:
        res[col] = res[col].astype(int)
    return res
