# to test and implement GPT-generated code
import numpy as np
import time

from src import config
from src.globals import get_afc, get_k_pv_dict, get_tt
from src.walk_time_dis import reject_outlier, map_platform_id_to_platform_uid, fit_transfer_time_dis_all, \
    map_path_seg_to_transfer_link


def test1():
    # res = fit_transfer_time_dis_all()
    res = map_path_seg_to_transfer_link()
    print(res.shape)
    print(res)


def test2():
    from src.globals import get_k_pv
    k_pv = get_k_pv()

    """
    [[1001100605 6 104291 1032 102320]
 [1001100705 6 104291 1032 102320]
 [1001100804 9 102320 1032 104291]
 ...
 [1136112502 9 104290 1032 102321]
 [1136112602 9 104290 1032 102321]
 [1136112702 9 104290 1032 102321]]
 """

    # print(k_pv[k_pv[:, -3] == "platform_swap"])
    print(k_pv[k_pv[:, 0] == 1001100705])
    print(k_pv[(k_pv[:, 0] >= 1001100700) & (k_pv[:, 0] <= 1001100709)])
    from src.utils import read_
    pa = read_(config.CONFIG["results"]["path"], show_timer=False)
    print(pa[(pa["path_id"] >= 1001100700) & (pa["path_id"] <= 1001100709)])


def test3():
    from src.globals import get_link_info, get_pl_info

    li = get_link_info()
    li = li[li[:, -1] != "in_vehicle"]
    print(li)

    print(li[(li[:, 1] == 1032) | (li[:, 2] == 1032)])

    platform_id1, platform_id2 = 102320, 104291
    li = get_link_info()
    li = li[li[:, -1] != "in_vehicle"]
    if li[(li[:, 1] == platform_id1) & (li[:, 2] == platform_id2)].shape[0] == 1:
        print("swap")
    uid = li[(li[:, 1] == platform_id1) & (li[:, -1] == "egress"), 2][0]
    print(uid)
    if li[(li[:, 1] == uid) & (li[:, 2] == platform_id2)].shape[0] == 1:
        print("egress-entry")
        pl_info = get_pl_info()


def test_computation_efficiency(tt, condition):
    # Measure execution time for the list comprehension approach
    start_time = time.time()
    trains_comprehension = [(tt[i, 0], tt[i, 5], tt[i + 1, 4]) for i in range(len(tt) - 1) if condition[i]]
    comprehension_time = time.time() - start_time

    # Measure execution time for the NumPy vectorized approach
    start_time = time.time()
    valid_indices = np.where(condition)[0]  # Get indices where condition is True
    trains_numpy = np.column_stack((tt[valid_indices, 0], tt[valid_indices, 5], tt[valid_indices + 1, 4])).tolist()
    numpy_time = time.time() - start_time

    # Measure execution time for the filtered array approach
    # start_time = time.time()
    # tt_filtered = tt[condition]  # Apply condition directly to the array
    # trains_filtered = np.column_stack((tt_filtered[:-1, 0], tt_filtered[:-1, 5], tt_filtered[1:, 4])).tolist()
    # filtered_time = time.time() - start_time

    # Print the time taken for each approach
    print(f"List Comprehension Time: {comprehension_time:.20f} seconds")
    print(f"NumPy Vectorized Time: {numpy_time:.20f} seconds")
    # print(f"Filtered Array Time: {filtered_time:.6f} seconds")

    # Return the results
    print(
        # trains_comprehension == trains_numpy,
        trains_comprehension[:10], trains_numpy[:10],
        len(trains_comprehension), len(trains_numpy),
        # len(trains_filtered)
    )
    return trains_comprehension, trains_numpy


if __name__ == '__main__':
    config.load_config()
    test1()
    # test2()
    # test3()

    # from src.utils import read_
    # df = read_(fn = config.CONFIG["results"]["assigned"], show_timer=False, latest_=True)
    # print(df)
    """
                   rid  iti_id     path_id  seg_id  train_id  board_ts  alight_ts
    0              132       1  1127110101       1  10200773     22740      23821
    1              284       1  1027103801       1  11002637     20292      21355
    4750           308       1  1132107101       1  10401755     22659      24066
    5302           312       1  1101103301       1  10200756     20219      22237
    5303           313       1  1101103301       1  10200756     20219      22237
    ...            ...     ...         ...     ...       ...       ...        ...
    109950037  2047044       1  1055100801       1  10402102     85925      86264
    109950038  2047046       1  1091111301       1  10401899     85983      86200
    109950039  2047049       1  1025111301       1  10401899     86095      86200
    109950040  2047059       1  1114105701       1  10702494     86172      86255
    109950041  2047060       1  1114105701       1  10702494     86172      86255
    """
    pass
