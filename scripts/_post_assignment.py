# after assignment plotting and analysis
from src import config
from src.timetable import plot_timetable_all


if __name__ == "__main__":

    config_file = "config3"
    config.load_config(f"configs/{config_file}.yaml")
    plot_timetable_all(save_subfolder=f"TT_{config_file}", separate_upd=True)
