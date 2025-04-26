# to test and implement GPT-generated code
import numpy as np
import time

from src import config
from src.globals import get_afc, get_k_pv_dict, get_tt
from src.walk_time_dis import *


def test1():
    # try egress
    path_to_eg_platform = map_path_id_to_platform(egress=True, entry=False)
    # print(path_to_eg_platform)

    platform_to_pl_id = map_platform_id_to_pl_id()
    # print(platform_to_pl_id)

    pl_id_to_x2pdf = map_pl_id_to_x2pdf_cdf(pdf=True, cdf=False)
    # print(pl_id_to_x2pdf)

    path_id = np.random.choice(get_k_pv()[:, 0], size=1)[0]
    print(path_id)

    x2pdf_for_this_path_id = pl_id_to_x2pdf[platform_to_pl_id[path_to_eg_platform[path_id]]]
    # print(x2pdf_for_this_path_id)

    egress_times = np.random.randint(low=0, high=500, size=10000)

    pdf_vals = cal_pdf(x2pdf_for_this_path_id, egress_times)
    print(pdf_vals)
    ...


def test2():
    # try entry
    path_to_en_platform = map_path_id_to_platform(egress=False, entry=True)
    platform_to_pl_id = map_platform_id_to_pl_id()
    pl_id_to_x2cdf = map_pl_id_to_x2pdf_cdf(pdf=False, cdf=True)
    path_id = np.random.choice(get_k_pv()[:, 0], size=1)[0]
    print(path_id)

    x2cdf_for_this_path_id = pl_id_to_x2cdf[platform_to_pl_id[path_to_en_platform[path_id]]]

    t_start = np.random.randint(low=0, high=250, size=10000)
    t_end = np.random.randint(low=t_start, high=500, size=10000)
    cdf_vals = cal_cdf(x2cdf_for_this_path_id, t_start, t_end)
    print(cdf_vals)

    ...


def test3():
    # try transfer
    ps2t = map_path_seg_to_platforms()

    t_to_x2cdf = map_transfer_link_to_x2cdf()

    assigned = read_("assigned", latest_=True, show_timer=False)
    _df = assigned[assigned["seg_id"] != 1].sample(n=1)
    path_id, seg_id = _df["path_id"].values[0], _df["seg_id"].values[0]
    print(path_id, seg_id - 1)

    x2cdf_for_this_path_seg = t_to_x2cdf[ps2t[(path_id, seg_id - 1)]]
    t_start = np.random.randint(low=0, high=250, size=10000)
    t_end = np.random.randint(low=t_start, high=500, size=10000)
    cdf_vals = cal_cdf(x2cdf_for_this_path_seg, t_start, t_end)
    print(cdf_vals)

    ...


if __name__ == '__main__':
    config.load_config()
    print("="*100)
    print("Test 1".center(100, " "))
    print("="*100)
    test1()
    print("="*100)
    print("Test 2".center(100, " "))
    print("="*100)
    test2()
    print("="*100)
    print("Test 3".center(100, " "))
    print("="*100)
    test3()

    pass
