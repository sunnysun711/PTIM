from src import config
from src.itinerary import attach_walk_dis_all
from src.utils import read_, save_
from src.globals import get_etd, get_ttd
from src.walk_time_dis_calculator import WalkTimeDisModel


def main():
    message = (
        '\033[33m'
        '======================================================================================\n'
        '[INFO] This script performs the following operations:\n'
        '         1. Loads the necessary data (left, ETD, TTD).\n'
        '         2. Calculates the distribution of all walk links in all left itineraries.\n'
        '         3. Saves the calculated distribution to the `dis.pkl` file.\n'
        '======================================================================================\n'
        '\033[0m'
    )
    print(message)
    
    # loading necessary data
    wtd = WalkTimeDisModel(get_etd(), get_ttd())  # build calculator
    left = read_(config.CONFIG["results"]["left"], show_timer=False)  # load to-calculate data
    
    # calculate distribution
    dis_df = attach_walk_dis_all(wtd=wtd, left=left)  # takes 35s to run (PC)
    save_(fn=config.CONFIG["results"]["dis"], data=dis_df, auto_index_on=False)  # only one time save
    return


if __name__ == "__main__":
    config.load_config()
    # main()
    pass

    
    