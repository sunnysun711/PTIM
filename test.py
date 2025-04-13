# to test and implement GPT-generated code

from src.utils import read_data_latest, ts2tstr
from scripts.analyze_egress import save_egress_times
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_egress_time_distribution(et: pd.DataFrame, title: str = ""):
    # Set up the figure and axes for the grid layout
    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 0.25])

    # Histogram of the egress time distribution (all day)
    ax_hist_right = fig.add_subplot(grid[0, 1])
    sns.histplot(y=et["egress_time"], kde=True, ax=ax_hist_right, color="blue", bins=30)
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
    ax_box.set_xticklabels([ts2tstr(ts) for ts in range(6 * 3600, 24 * 3600 + 1, _bin_width)])
    ax_box.set_xlim(- 0.5, (24 - 6) * 3600 // _bin_width - 0.5)
    for label in ax_box.get_xticklabels():
        label.set_rotation(90)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # save_egress_times(save_fn="egress_times")
    et = read_data_latest(fn="egress_times")
    print(et.sample(n=20))

    gb = et.groupby(["node1", "node2"])
    for (node1, node2), group in gb:
        print(node1, node2, group.shape)
        plot_egress_time_distribution(group, title=f"{node1} -> {node2}")

    pass
