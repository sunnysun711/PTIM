import yaml
import os



if __name__ == '__main__':
    from src import config
    config.load_config()
    from src.walk_time_dis import get_egress_link_groups
    from src.utils import read_, save_, get_latest_file_index, get_file_path

    # print(get_file_path("assigned"))
    # print(read_(fn="assigned.pkl", latest_=True))
    # assigned = read_(fn="assigned.pkl", latest_=True)
    # save_(fn="assigned", data=assigned, auto_index_on=True)
    # pla = read_("platform.json")
    print(get_egress_link_groups())
    pass
