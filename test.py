# to test and implement GPT-generated code

import matplotlib.pyplot as plt
from src import config
config.load_config()
from src.utils import ts2tstr, read_
from scripts.analyze_egress import save_egress_times
import pandas as pd
import seaborn as sns
import matplotlib

from src.walk_time_dis import get_egress_link_groups

matplotlib.use('TkAgg')


def plot_egress_time_distribution(et: pd.DataFrame, title: str = ""):
    # Set up the figure and axes for the grid layout
    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 0.25])

    # Histogram of the egress time distribution (all day)
    ax_hist_right = fig.add_subplot(grid[0, 1])
    sns.histplot(y=et["egress_time"], kde=True,
                 ax=ax_hist_right, color="blue", bins=30)
    ax_hist_right.set_xlabel("Frequency")
    ax_hist_right.set_ylabel("")

    # Scatter plot in the center
    ax_scatter = fig.add_subplot(grid[0, 0], sharey=ax_hist_right)
    ax_scatter.scatter(et['alight_ts'], et["egress_time"], alpha=0.4)
    ax_scatter.set_xlabel("Alight Timestamp")
    ax_scatter.set_ylabel("Egress Time")
    ax_scatter.set_xticks(range(6 * 3600, 24 * 3600 + 1, 3600))
    ax_scatter.set_xticklabels([f"{i:02}:00" for i in range(6, 25, 1)])
    ax_scatter.set_xlim(6 * 3600, 24 * 3600)
    ax_scatter.set_title("Egress Time Distribution " + title)

    # Boxplot of egress time versus alight_ts, with customized bin width
    _bin_width = 1800
    et['alight_ts_binned'] = (et['alight_ts'] // _bin_width) * _bin_width

    ax_box = fig.add_subplot(grid[1, 0])
    sns.boxplot(x="alight_ts_binned", y="egress_time", data=et, ax=ax_box)
    ax_box.set_xlabel(f"Alight Timestamp")
    ax_box.set_ylabel("Egress Time")
    xticks = [i - 0.5 for i in range((24 - 6) * 3600 // _bin_width + 1)]
    ax_box.set_xticks(xticks)
    ax_box.set_xticklabels([ts2tstr(ts)
                           for ts in range(6 * 3600, 24 * 3600 + 1, _bin_width)])
    ax_box.set_xlim(- 0.5, (24 - 6) * 3600 // _bin_width - 0.5)
    for label in ax_box.get_xticklabels():
        label.set_rotation(90)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_egress_time_distribution2(et: pd.DataFrame, title: str = ""):
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Generate two groups of normal distribution data
    np.random.seed(42)
    data1 = np.random.normal(loc=10, scale=2, size=300)
    data2 = np.random.normal(loc=20, scale=3, size=200)
    combined_data = np.concatenate([data1, data2])

    # Perform KDE
    kde = gaussian_kde(combined_data)

    # Define the range for the variable
    x_min, x_max = 5, 25  # Limit the range to [5, 25]
    x_values = np.linspace(x_min, x_max, 1000)

    # Compute PDF
    pdf_values = kde(x_values)

    # Compute CDF using numerical integration
    cdf_values = [quad(kde, x_min, x)[0] for x in x_values]

    # Plot PDF
    plt.figure(figsize=(10, 6))
    sns.histplot(combined_data, kde=False, bins=30,
                 stat='density', label='Histogram')
    plt.plot(x_values, pdf_values, label='PDF (KDE)', color='red')
    plt.title('PDF with Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot CDF
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, cdf_values, label='CDF', color='blue')
    plt.title('CDF from Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.show()

    # Print bandwidth
    print(f"Bandwidth of KDE: {kde.factor}")
    return


if __name__ == '__main__':
    # save_egress_times(save_fn="egress_times")

    et_ = read_(fn="egress_times_1", show_timer=False, latest_=False).set_index(["node1", "node2"])
    # todo: reject outliers of egress times.
    # print(et.sample(n=20))

    # gb = et.groupby(["node1", "node2"])
    # for (node1, node2), group in gb:
    #     print(node1, node2, group.shape)
    #     plot_egress_time_distribution(group, title=f"{node1} -> {node2}")

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from scipy.stats import gaussian_kde

    res = get_egress_link_groups()
    for uid, platform_links in res.items():
        print(uid)
        for egress_links in platform_links:
            # all egress links on the same platform
            et = et_[et_.index.isin(egress_links)].copy()
            print(et.shape, egress_links)
            plot_egress_time_distribution(et, title=f"{egress_links}")

    pass
