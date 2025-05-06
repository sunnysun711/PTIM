import argparse


def main(config_file):
    # Importing the necessary modules
    from src import config

    # Load the configuration using the config file path
    config.load_config(config_file)

    # Running the various modules
    if not config.CONFIG["use_existing"]["network"]:
        from scripts import prep_network
        prep_network.main()

    if not config.CONFIG["use_existing"]["itinerary"]:
        from scripts import find_feas_iti
        find_feas_iti.main()

    if not config.CONFIG["use_existing"]["trajectory"]:
        from scripts import split_feas_iti
        split_feas_iti.main()

    if not config.CONFIG["use_existing"]["walk_times"]:
        from scripts import analyze_walk_time
        analyze_walk_time.main()

    from scripts import calculate_distribution
    calculate_distribution.main()
    
    from scripts import assign_traj
    assign_traj.main()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run the script with a given config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,  # Optional argument to allow defaults
        default="configs/config3.yaml",  # Default config path
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Pass the config file to the main function
    main(args.config)