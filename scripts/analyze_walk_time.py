from src import config
from src.utils import save_, read_
from src.walk_time_filter import filter_egress_all, filter_transfer_all


def save_filtered_walk_times():
    eg_t = filter_egress_all()
    tr_t = filter_transfer_all()
    save_(fn=config.CONFIG["results"]["egress_times"],
          data=eg_t, auto_index_on=True)
    save_(fn=config.CONFIG["results"]["transfer_times"],
          data=tr_t, auto_index_on=True)
    ...


def main():
    """
    Main function to save filtered walk times and physical links information.
    """
    # Save filtered walk times
    save_filtered_walk_times()

    # Plot egress times
    ...

    # Plot transfer times
    ...

    # Save distributions: ETD, TTD
    ...
    ...


if __name__ == "__main__":
    config.load_config()
    main()
    ...
