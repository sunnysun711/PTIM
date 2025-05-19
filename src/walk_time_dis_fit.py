"""
This module contains utility functions for fitting probability density 
functions (PDF) and cumulative distribution functions (CDF) to various time
data, including egress and transfer times.

Key functionalities include:
- Rejecting outliers based on the z-score or boxplot method.
- Fitting different distribution models (KDE, Gamma, Log-Normal) to egress 
    and transfer time data.
- Evaluating the goodness-of-fit using the Kolmogorov-Smirnov (K-S) test.

Functions:
    reject_outlier_bd(): Calculate the bounds for outlier rejection based on 
        a specified method.
    reject_outlier(): Reject outliers from data using the method specified 
        (z-score or boxplot).
    fit_pdf_cdf(): Fit a probability distribution (KDE, Gamma, or Log-Normal) 
        to the input data.
    evaluate_fit(): Perform a K-S test to evaluate the fit of a CDF to data.
    fit_one_physical_platform(): Fit the distribution of egress times for a 
        single physical platform.
    fit_platform_egress_time_dis_all_parallel(): Fit egress time 
        distributions for all physical platforms in parallel.
    fit_transfer_time_dis_all(): Fit transfer time distributions for all 
        transfer links.
"""
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
        
        - boxplot: "https://www.secrss.com/articles/11994"
        - zscore: "https://www.zhihu.com/question/38066650"

    :param data: Input data array.
    :type data: np.ndarray

    :param method: Outlier detection method ('zscore' or 'boxplot').
    :type method: str, optional, default="zscore"

    :param abs_max: Absolute maximum value constraint.
    :type abs_max: int | None, optional, default=500

    :returns: A tuple of lower and upper bounds for valid data.
    :rtype: tuple[float, float]

    :raises Exception: If an invalid method is provided.
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
        
        - boxplot: "https://www.secrss.com/articles/11994"
        - zscore: "https://www.zhihu.com/question/38066650"
        
    :param data: Input data array.
    :type data: np.ndarray

    :param method: Outlier detection method ('zscore' or 'boxplot').
    :type method: str, optional, default="zscore"

    :param abs_max: Absolute maximum value constraint.
    :type abs_max: int, optional, default=500

    :returns: Cleaned data array.
    :rtype: np.ndarray

    :raises Exception: If an invalid method is provided.
    """
    lb, ub = reject_outlier_bd(data, method=method, abs_max=abs_max)
    return data[(data >= lb) & (data <= ub)]


def fit_pdf_cdf(data: np.ndarray, method: str = "kde", hbar: float = None) -> tuple[Callable, Callable]:
    """
    Fit a probability density function (PDF) and cumulative distribution function (CDF) to the input data.

    :param data: Input data array.
    :type data: np.ndarray

    :param method: Method to fit the PDF and CDF. Options are 'kde' (Kernel Density Estimation),
                   'gamma' (Gamma Distribution), or 'lognorm' (Log-Normal Distribution).
    :type method: str, optional, default="kde"

    :param hbar: If provided, indicates that data = true_walk + Uniform(0, hbar),
                 and a deconvolution step will be applied to recover true_walk distribution.
    :type hbar: float or None

    :returns: A tuple containing the fitted PDF function and CDF function.
    :rtype: tuple[Callable, Callable]

    :raises Exception: If an invalid method is provided.
    """
    import numpy as np
    from scipy.stats import gaussian_kde, gamma, lognorm
    from scipy.fft import fft, ifft

    # Evaluation grid
    # x = np.linspace(0, np.max(data) * 1.2, 2**12)
    # dx = x[1] - x[0]
    x = np.arange(0, 501)
    dx = 1.0

    # Step 1: Fit PDF and CDF (without deconvolution)
    if method == "kde":
        kde = gaussian_kde(data)
        raw_pdf = kde(x)
    elif method == "gamma":
        params = gamma.fit(data, floc=0)
        raw_pdf = gamma(*params).pdf(x)
    elif method == "lognorm":
        params = lognorm.fit(data)
        raw_pdf = lognorm(*params).pdf(x)
    else:
        raise Exception("Please use either 'kde', 'gamma', or 'lognorm' method to fit pdf!")

    # Step 2: Deconvolution if hbar is specified
    if hbar is not None:
        # Construct uniform kernel in [0, hbar]
        kernel = np.zeros_like(x)
        mask = (x >= 0) & (x <= hbar)
        kernel[mask] = 1 / hbar

        F_pdf = fft(raw_pdf)
        F_kernel = fft(kernel)
        eps = 1e-3  # regularization
        F_deconv = F_pdf / (F_kernel + eps)
        deconv_pdf = np.real(ifft(F_deconv))
        deconv_pdf = np.clip(deconv_pdf, 0, None)
        deconv_pdf /= np.sum(deconv_pdf) * dx  # re-normalize
        pdf_vals = deconv_pdf
    else:
        pdf_vals = raw_pdf

    # Step 3: Build interpolation-based PDF and CDF functions
    def pdf_fn(xx): return np.interp(xx, x, pdf_vals, left=0, right=0)
    cdf_vals = np.cumsum(pdf_vals) * dx
    def cdf_fn(xx): return np.interp(xx, x, cdf_vals, left=0, right=1)

    return pdf_fn, cdf_fn


def evaluate_fit(data: np.ndarray, cdf_func: Callable, hbar: float = None) -> tuple[float, float]:
    """
    Evaluate the fit of a CDF function to empirical data using the Kolmogorov-Smirnov test.

    :param data: Input data array.
    :type data: np.ndarray
    
    :param cdf_func: Fitted CDF function.
    :type cdf_func: Callable
    
    :param hbar: If given, compare to Uniform(0, hbar) + fitted CDF for deconvolution.
    :type hbar: float or None

    :return: KS statistic and p-value.
    """
    data_sorted = np.sort(data)
    if hbar is None:
        return kstest(data_sorted, cdf_func)
    else:
        # simulate uniform wait and fitted walk sum
        u = np.random.uniform(0, hbar, size=data_sorted.shape[0])
        walk_sim = np.random.choice(data_sorted, size=data_sorted.shape[0])
        sim_sum = u + walk_sim
        return kstest(sim_sum, cdf_func)


def fit_one_physical_platform(pp_id: int, eg_t_data: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """
    Fit the distribution of physical platform egress time.

    :param pp_id: Physical platform ID.
    :type pp_id: int

    :param eg_t_data: Egress time numpy array of the current physical platform.
    :type eg_t_data: np.ndarray

    :param x: X values for PDF and CDF, usually [0, 500] with 501 points.
    :type x: np.ndarray

    :returns: An array with shape (x.size, 14) or None if no data is available.
    :rtype: np.ndarray | None
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
    Fit the distribution of physical platform egress time in parallel.

    :param eg_t: Egress time dataframe.
        Each row is [rid (index), physical_platform_id, alight_ts, egress_time].
    :type eg_t: pd.DataFrame

    :param n_jobs: Number of threads to use. Default is -1, which uses all available threads.
    :type n_jobs: int, optional, default=-1

    :returns: A DataFrame with the fitted distribution for each physical platform.
        
        Each row contains the following columns:

            - `pp_id`: Physical platform ID.
            - `x`: x values (0-500).
            - `kde_pdf`: KDE fitted probability density function.
            - `kde_cdf`: KDE fitted cumulative distribution function.
            - `kde_ks_stat`: K-S statistic for KDE fit.
            - `kde_ks_p_value`: K-S p-value for KDE fit.
            - `gamma_pdf`: Gamma fitted probability density function.
            - `gamma_cdf`: Gamma fitted cumulative distribution function.
            - `gamma_ks_stat`: K-S statistic for Gamma fit.
            - `gamma_ks_p_value`: K-S p-value for Gamma fit.
            - `lognorm_pdf`: Log-normal fitted probability density function.
            - `lognorm_cdf`: Log-normal fitted cumulative distribution function.
            - `lognorm_ks_stat`: K-S statistic for Log-normal fit.
            - `lognorm_ks_p_value`: K-S p-value for Log-normal fit.

    :rtype: pd.DataFrame
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
    :param tr_t: Transfer time dataframe with columns: 
        
        [rid (index), path_id, seg_id, pp_id1, pp_id2, alight_ts, transfer_time, transfer_type]
    :type tr_t: pd.DataFrame

    :return: Dataframe with columns:
        [pp_id_min, pp_id_max, x, kde_cdf, gamma_cdf, lognorm_cdf]
        
        where: 
            
        - `pp_id_min` and `pp_id_max` are the physical platform ids of the 
            transfer link,
        - `x` is the transfer time, and kde_cdf, gamma_cdf, lognorm_cdf are the 
            CDF values of the transfer time distribution fitted with KDE, 
            Gamma, and Log-Normal distributions respectively.
        - Each row is a transfer link.
    :rtype: pd.DataFrame
    
    Note: 
        
        - for platform_swap transfers (pp_id_min=pp_id_max), the CDF values are all 1, and x is only one value as 0.
        - for egress-entry transfers, the CDF values are normalized to [0, 1].
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


def _test_fit_conv(hbar: float = None):
    """
    Test the distribution fitting (with optional deconvolution) on sample transfer time data.
    """
    from src.utils import read_
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kstest
    print("[TEST] Load sample transfer data...")
    # [rid(index), path_id, seg_id, pp_id1, pp_id2, alight_ts, transfer_time, transfer_type]
    df = read_("transfer_times", show_timer=False, latest_=True)

    pp_id1, pp_id2 = df[df['transfer_type'] == "egress-entry"].sample(n=1)[["pp_id1", "pp_id2"]].values.flatten()
    print(pp_id1, pp_id2)
    
    df_sub = df[(df['pp_id1'] == pp_id1) & (df['pp_id2'] == pp_id2)]
    df_sub = df_sub[df_sub["transfer_time"] <= 500]

    sample_data = df_sub["transfer_time"].values
    x = np.arange(0, 501)

    methods = ["kde", "gamma", "lognorm"]
    results_pdf = {}
    results_cdf = {}
    results_pdf2 = {}
    results_cdf2 = {}

    for method in methods:
        print(f"\n[INFO] Fitting method: {method} {'(with deconvolution)' if hbar else ''}")
        pdf, cdf = fit_pdf_cdf(sample_data, method=method, hbar=hbar)
        pdf_vals = pdf(x)
        cdf_vals = cdf(x)
        stat, pval = evaluate_fit(sample_data, cdf, hbar=hbar)
        print(f"  KS statistic: {stat:.4f}, p-value: {pval:.4f}")
        results_pdf[method] = pdf_vals
        results_cdf[method] = cdf_vals
    
    for method in methods:
        print(f"\n[INFO] Fitting method: {method} {'(with deconvolution)' if hbar else ''}")
        pdf, cdf = fit_pdf_cdf(sample_data, method=method, hbar=None)
        pdf_vals = pdf(x)
        cdf_vals = cdf(x)
        stat, pval = evaluate_fit(sample_data, cdf, hbar=None)
        print(f"  KS statistic: {stat:.4f}, p-value: {pval:.4f}")
        results_pdf2[method] = pdf_vals
        results_cdf2[method] = cdf_vals

    # Plot PDF comparison with histogram
    plt.figure(figsize=(8, 5))
    plt.hist(sample_data, bins=50, density=True, alpha=0.4, label="Empirical (hist)", color='gray')
    for method in methods:
        label = f"{method.upper()} {'(deconv)' if hbar else ''}"
        plt.plot(x, results_pdf[method], label=label, lw=2)
    
    for method in methods:
        label = f"{method.upper()}"
        plt.plot(x, results_pdf2[method], label=label+"(no H)", lw=2)
    
    plt.title(f"PDF Comparison {'(Deconvolved)' if hbar else '(Raw)'}")
    plt.xlabel("Transfer time (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return

if __name__ == "__main__":
    from src import config
    config.load_config("configs/config_backup.yaml")
    
    _test_fit_conv(hbar=500.0)
    pass