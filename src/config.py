import os
import yaml

# Global variable to store the configuration (could also use a constant if needed)
# To use the load_config update, make sure to access it with config.CONFIG instead of importing CONFIG directly.
CONFIG: dict = {}


def check_data_presence(config: dict):
    """Check if the required data files exist in the data folder."""
    files = os.listdir(config['data_folder'])
    for data_file in config['data']:
        if data_file not in files:
            raise FileNotFoundError(f"File {data_file} not found in {config['data_folder']}")


def gen_results_folders(config: dict):
    """Generate necessary result folders."""
    results_folder = config['results_folder']
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"[INFO] Created results folder: {results_folder}")

    for subfolder in config['results_subfolder']:
        subfolder_path = os.path.join(results_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"[INFO] Created subfolder: {subfolder_path}")


def gen_figure_folders(config: dict):
    """Generate necessary figure folders."""
    figure_folder = config['figure_folder']
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        print(f"[INFO] Created figure folder: {figure_folder}")


def check_used_results(config: dict):
    """Check if required result files exist."""
    if config['use_existing']["network"]:
        check_file_exists(config, config["results_subfolder"]["network"], "node")
        check_file_exists(config, config["results_subfolder"]["network"], "link")

    if config['use_existing']["path"]:
        check_file_exists(config, config["results_subfolder"]["path"], "path")
        check_file_exists(config, config["results_subfolder"]["path"], "pathvia")

    if config['use_existing']["itinerary"]:
        check_file_exists(config, config["results_subfolder"]["itinerary"], "feas_iti")

    if config['use_existing']["trajectory"]:
        check_file_exists(config, config["results_subfolder"]["trajectory"], "left")
        check_file_exists(config, config["results_subfolder"]["trajectory"], "stashed")


def check_file_exists(config: dict, subfolder: str, result_key: str):
    """Helper function to check if a result file exists."""
    result_file = config['results'][result_key]
    file_path = os.path.join(config['results_folder'], subfolder, result_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{subfolder} results `{result_file}` not found.")


def load_config(config_file='configs/config1.yaml') -> dict:
    """Load configuration from the YAML file and perform checks."""
    global CONFIG
    with open(config_file, 'r') as f:
        CONFIG = yaml.safe_load(f)

    check_data_presence(CONFIG)
    gen_results_folders(CONFIG)
    check_used_results(CONFIG)
    gen_figure_folders(CONFIG)

    return CONFIG
