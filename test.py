# to test and implement GPT-generated code
import numpy as np
import time

from src import config


def test1():
    from src.walk_time_dis import get_x2pdf, get_pdf, node_id_to_pl_id
    from src.globals import get_k_pv

    k_pv = get_k_pv()
    # print(k_pv)

    k_pv_prev = k_pv[:-1, :]
    k_pv_next = k_pv[1:, :]
    transfers = np.where(
        (k_pv_prev[:, 0] == k_pv_next[:, 0]) & (k_pv_prev[:, 4] == "egress") & (k_pv_next[:, 4] == "entry"))

    transfers_egress = k_pv_prev[transfers][:, [0, 1, 2, 3]]
    transfers_entry = k_pv_next[transfers][:, [3]]
    print(transfers_egress, transfers_entry)
    transfer = np.hstack((transfers_egress, transfers_entry))  # [path_id, pv_id, platform_1, uid, platform_2]
    transfer = transfer[transfer[:, -2] == 1032]  # 1010: huaishudian 1028: Tianfu Square
    print(transfer)

    # random select
    path_id, pv_id, node1, uid, node2 = transfer[np.random.randint(0, transfer.shape[0])]
    print(node1, node2)

    pl_id1 = node_id_to_pl_id(node_id=node1)
    pl_id2 = node_id_to_pl_id(node_id=node2)
    print(pl_id1, pl_id2)

    # path_id = np.random.choice(k_pv[k_pv[:, 1] > 3][:, 0])
    # print(k_pv[k_pv[:, 0] == path_id])

    x_range = np.arange(0, 501)

    pdf_X = get_pdf(pl_id1, x_range)
    pdf_Y = get_pdf(pl_id2, x_range)

    pdf_Z = np.zeros(1001)  # Z的取值范围是0到1000

    # 离散卷积
    for z in range(1001):
        # pdf_Z[z] = np.sum(pdf_X * np.interp(z - x_range, x_range, pdf_Y, left=0, right=0))
        # pdf_Z[z] = np.sum(pdf_X * (pdf_Y[z - x_range] if (0 <= z - x_range) else 0))
        valid_indices = (z - x_range >= 0) & (z - x_range <= 500)  # 确保索引有效
        pdf_Z[z] = np.sum(pdf_X[valid_indices] * pdf_Y[z - x_range[valid_indices]])

    print(pdf_Z)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(0, 1001), pdf_Z, label="PDF of X + Y")
    plt.plot(np.arange(0, 501), pdf_X, label="PDF of X")
    plt.plot(np.arange(0, 501), pdf_Y, label="PDF of Y")
    plt.xlabel('z = X + Y')
    plt.ylabel('f_Z(z)')
    plt.title('PMF of X + Y')
    plt.legend()
    plt.grid(True)
    plt.show()


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


if __name__ == '__main__':
    config.load_config()
    # test1()
    # test2()
    # test3()
    from src.utils import read_
    df = read_(fn = config.CONFIG["results"]["assigned"], show_timer=False, latest_=True)
    print(df)
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
