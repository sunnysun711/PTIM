import os
import time

import networkx as nx
import numpy as np
import pandas as pd

from src.metro_net import *
from src.utils import read_data
from src import DATA_DIR


def test_k_paths():
    nodes = read_data("node_info", show_timer=False)
    links = read_data("link_info", show_timer=False)
    # nodes = gen_node_from_sta()
    # links = gen_links(platform_swap_time=20, entry_time=60, egress_time=60)
    net = ChengduMetro(nodes, links)

    source, target = np.random.choice(range(1001, 1137), 2)
    # source, target = 1099, 1078  # 1号线内部换乘
    # source, target = 1050, 1131  # 多条 有效路径，用于检验yen算法的求解效果
    # source, target = 1088, 1118  # 西南财大 -> 骡马市 -> 火车南站 -> 神仙树
    # source, target = 1131, 1029
    sta_dict = pd.read_pickle(os.path.join(DATA_DIR, "STA.pkl")) \
        .drop_duplicates(subset="STATION_UID") \
        .reset_index().set_index("STATION_UID")['STATION_NAME'].to_dict()
    print(sta_dict)
    print(sta_dict[source], " -> ", sta_dict[target])

    length, s_path = nx.single_source_dijkstra(net.G, source=source, target=target)
    print(length, s_path)

    max_length = min(length * 1.6, length + 600)
    print(max_length)

    lens, paths = net.find_k_paths_via_yen(
        shortest_path=s_path, shortest_path_length=length, max_length=max_length
    )
    for le, pa in zip(lens, paths):
        print(le, net.compress_passing_info(path=pa))

    return


def main():
    a = time.time()
    nodes = read_data("node_info", show_timer=False)
    links = read_data("link_info", show_timer=False)
    # nodes = gen_node_from_sta()
    # links = gen_links(platform_swap_time=20, entry_time=60, egress_time=60)
    net = ChengduMetro(nodes, links)
    # print(net.G.edges[102360, 102370])
    df_p, df_pv = net.find_all_pairs_k_paths()
    print(time.time() - a)

    print(df_p)
    print(df_pv)
    # net.plot_metro_net(coordinates=read_data("coordinates.csv"))

    pass


if __name__ == '__main__':
    main()
