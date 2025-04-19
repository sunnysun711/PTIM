# to test and implement GPT-generated code
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kstest

matplotlib.use('TkAgg')


def calculate_kde_fit_quality(data, kde, x_values):
    """
    Calculate the Mean Squared Error (MSE) between the KDE fit and the histogram of the data.

    Parameters:
    -----------
    data : numpy.ndarray
        The raw data points.
    kde : gaussian_kde object
        The KDE object that is fitted to the data.
    x_values : numpy.ndarray
        The x-values for which the KDE and histogram will be compared.
    num_bins : int, optional (default=30)
        The number of bins for the histogram.

    Returns:
    --------
    mse : float
        The Mean Squared Error between the KDE and the histogram.
    """
    # Compute the histogram of the data
    hist, bin_edges = np.histogram(data, bins=50, range=(x_values.min(), x_values.max()), density=True)

    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Interpolate the KDE values at the bin centers
    kde_values = kde(bin_centers)

    # Calculate MSE between KDE and histogram
    mse = np.mean((hist - kde_values) ** 2)

    return mse


def evaluate_fit(data: np.ndarray, pdf_func: Callable, cdf_func: Callable) -> dict:
    # 计算经验分布函数 (ECDF)
    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)

    # 使用K-S检验来评估拟合效果
    ks_stat, ks_p_value = kstest(data_sorted, cdf_func)

    # 返回K-S统计量和P值
    return {"ks_stat": ks_stat, "ks_p_value": ks_p_value}


def plot_egress_time_distribution1():
    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src import config
    from src.utils import read_, execution_timer
    from src.walk_time_dis import get_egress_link_groups, get_reject_outlier_bd

    config.load_config()

    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False)
    et__ = et_.set_index(["node1", "node2"])
    uid = np.random.choice(range(1001, 1137), size=1)[0]
    # uid = 1033
    # uid = 1065
    # uid = 1064
    platform_links = get_egress_link_groups(et_=et_)[uid]
    print(uid, platform_links)


    for egress_links in platform_links:
        # all egress links on the same platform
        et = et__[et__.index.isin(egress_links)]
        if et.shape[0] == 0:
            continue
        title = f"{uid}_{[i[0] for i in egress_links]}"
        raw_size = et.shape[0]

        lb, ub = get_reject_outlier_bd(data=et['egress_time'].values, method="zscore", abs_max=500)
        et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
        print(f"{title} | Data size: {et.shape[0]}/{raw_size} | BD: [{lb:.4f}, {ub:.4f}]")

        data = et['egress_time'].values

        # Perform KDE
        from src.walk_time_dis import fit_pdf_cdf
        methods = ["kde", "lognorm", "gamma"]
        x_values = np.linspace(0, 500, 501)

        fit_results: dict = {"uid": uid, "x": x_values}
        for met in methods:
            pdf_f, cdf_f = fit_pdf_cdf(data, method=met)
            print(f"Method: {met}")
            # Compute PDF  ~ 0.4 s
            pdf_values = pdf_f(x_values)

            # Compute CDF using numerical integration
            cdf_values = cdf_f(x_values)
            cdf_values = cdf_values / cdf_values[-1]  # scale to [0, 1] for some not-fit-well data

            kstest = evaluate_fit(data=data, pdf_func=pdf_f, cdf_func=cdf_f)
            # fit_results.append([x_values, pdf_values, cdf_values, kstest["ks_stat"], kstest["ks_p_value"]])
            fit_results[f"pdf_{met}"] = pdf_values
            fit_results[f"cdf_{met}"] = cdf_values
            fit_results[f"ks_stat_{met}"] = kstest["ks_stat"]
            fit_results[f"ks_p_value_{met}"] = kstest["ks_p_value"]
            print(fit_results)

        print(pd.DataFrame(fit_results))

        # Plot PDF
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=False, bins=30, stat='density', label='Histogram')
        plt.plot(x_values, pdf_values, label='PDF (KDE)', color='red')

        # Create a secondary y-axis for the CDF
        ax2 = plt.gca().twinx()
        ax2.plot(x_values, cdf_values, label='CDF', color='green')
        ax2.set_ylabel('CDF')
        ax2.set_ylim((0, 1))
        # ax2.legend(loc='upper right')
        #
        plt.title('PDF with Kernel Density Estimation')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

        # 因为数据精度为1秒，所以需要：
        # 1. 衡量当前kde的拟合效果是否满足需求
        # 2. （不满足）调整kde的带宽参数或采用其他方法，使得kde的拟合效果满足要求
        # 3. （满足）以1秒为单位，计算每个时间点的pdf值，并保存在: ETD_{KDE, GAMMA, LOGNORM}_1.pkl.中，index: uid，columns: 0-500。
    return


def plot_egress_time_distribution2():
    from scipy.stats import gaussian_kde, kstest, gamma, lognorm
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src import config
    from src.utils import read_, execution_timer
    from src.walk_time_dis import get_egress_link_groups, get_reject_outlier_bd, fit_pdf_cdf

    config.load_config()

    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False)
    et__ = et_.set_index(["node1", "node2"])
    uid = np.random.choice(range(1001, 1137), size=1)[0]
    # uid = 1033
    # uid = 1065
    # uid = 1064
    platform_links = get_egress_link_groups(et_=et_)[uid]
    print(uid, platform_links)

    for egress_links in platform_links:
        et = et__[et__.index.isin(egress_links)]
        if et.shape[0] == 0:
            continue
        title = f"{uid}_{[i[0] for i in egress_links]}"
        raw_size = et.shape[0]

        lb, ub = get_reject_outlier_bd(data=et['egress_time'].values, method="zscore", abs_max=500)
        et = et[(et['egress_time'] >= lb) & (et['egress_time'] <= ub)]
        print(f"{title} | Data size: {et.shape[0]}/{raw_size} | BD: [{lb:.4f}, {ub:.4f}]")

        data = et['egress_time'].values

        # Methods to fit: KDE, LogNormal, Gamma
        methods = ["kde", "lognorm", "gamma"]
        x_values = np.linspace(0, 500, 501)

        # Prepare for plotting
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax2 = ax.twinx()

        # Loop through methods
        ks_results = {}
        for method in methods:
            pdf_f, cdf_f = fit_pdf_cdf(data, method=method)

            # Calculate PDF and CDF values
            pdf_values = pdf_f(x_values)
            cdf_values = cdf_f(x_values)
            cdf_values = cdf_values / cdf_values[-1]  # scale CDF to [0, 1]

            # Plot PDF
            ax.plot(x_values, pdf_values, label=f'{method} PDF', linestyle='-', lw=2)

            # Plot CDF with a secondary y-axis
            ax2.plot(x_values, cdf_values, label=f'{method} CDF', linestyle='--', lw=2)
            ax2.set_ylim((0, 1))

            # Perform K-S test and store results
            ks_stat, ks_p_value = kstest(data, cdf_f)
            print(f"{method.capitalize():<10} K-S Test - Statistic: {ks_stat:.4f}, P-value: {ks_p_value:.4f}")

        # Plot Histogram
        sns.histplot(data, kde=False, ax=ax, bins=30, stat='density', label='Histogram', alpha=0.5)

        ax2.set_ylabel('CDF')
        ax2.legend(loc='upper right')

        ax.set_title(f'{uid} Egress Time Distribution with Various Fits')
        ax.set_xlabel('Egress Time')
        ax.set_ylabel('Density')
        ax.set_xlim((0, data.max()))
        ax.legend(loc='lower right')

        # Show plot
        plt.show()


if __name__ == '__main__':
    # plot_egress_time_distribution2()
    plot_egress_time_distribution1()
    pass
