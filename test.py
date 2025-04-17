# to test and implement GPT-generated code

import matplotlib
import numpy as np

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


def plot_egress_time_distribution2():
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
        pdf_f, cdf_f = fit_pdf_cdf(data, method="lognorm")

        x_values = np.linspace(0, 500, 501)

        # Compute PDF  ~ 0.4 s
        pdf_values = pdf_f(x_values)

        # Compute CDF using numerical integration
        cdf_values = cdf_f(x_values)
        cdf_values = cdf_values / cdf_values[-1]  # scale to [0, 1] for some not-fit-well data

        import pandas as pd
        res = pd.DataFrame({
            "uid": [uid] * len(x_values),
            "x": x_values,
            "pdf": pdf_values,
            "cdf": cdf_values
        })
        print(res.head(20))
        print(res.tail(20))

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

        # Calculate KDE fit quality
        mse = calculate_kde_fit_quality(data, pdf_f, x_values)  # todo: fix bugs always zero
        print(f"KDE Fit Quality (MSE): {mse:.4f}")

        # print(f"Bandwidth of KDE: {pdf_f.factor}")

        # todo: 因为数据精度为1秒，所以需要：
        # 1. 衡量当前kde的拟合效果是否满足需求
        # 2. （不满足）调整kde的带宽参数或采用其他方法，使得kde的拟合效果满足要求
        # 3. （满足）以1秒为单位，计算每个时间点的pdf值，并保存在: etd_1.pkl.中，index是uid，columns就是1-500秒。
    return


if __name__ == '__main__':
    plot_egress_time_distribution2()
    pass
