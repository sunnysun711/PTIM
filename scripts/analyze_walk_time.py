from src import config
from src.utils import save_, read_, pd
from src.walk_time_filter import filter_egress_all, filter_transfer_all
from src.walk_time_plot import plot_egress_all, plot_transfer_all


def save_filtered_walk_times() -> tuple[pd.DataFrame, pd.DataFrame]:
    eg_t = filter_egress_all()
    tr_t = filter_transfer_all()
    save_(fn=config.CONFIG["results"]["egress_times"],
          data=eg_t, auto_index_on=True)
    save_(fn=config.CONFIG["results"]["transfer_times"],
          data=tr_t, auto_index_on=True)
    return eg_t, tr_t


def main():
    """
    Main function to save filtered walk times and physical links information.
    """
    # Save filtered walk times
    eg_t, tr_t = save_filtered_walk_times()

    # Plot egress times
    plot_egress_all(eg_t, save_on=True, save_subfolder="ETD0")

    # Plot transfer times
    plot_transfer_all(tr_t, save_on=True, save_subfolder="TTD0")

    # Save distributions: ETD, TTD
    ...
    ...


if __name__ == "__main__":
    config.load_config()
    main()
    ...
