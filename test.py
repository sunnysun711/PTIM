import os
import networkx as nx
import numpy as np
import pandas as pd

from src.metro_net import *
from src.utils import read_data
from src import DATA_DIR


def main():
    nodes = read_data("node_info", show_timer=False)
    links = read_data("link_info", show_timer=False)
    # nodes = gen_node_from_sta()
    # links = gen_links(platform_swap_time=20, entry_time=120, egress_time=120)
    net = ChengduMetro(nodes, links)

    # net.plot_metro_net(coordinates=read_data("coordinates.csv"))

    source, target = np.random.choice(range(1001, 1137), 2)
    # source, target = 1099, 1078  # 1号线内部换乘
    # source, target = 1050, 1131  # 多条 有效路径，用于检验yen算法的求解效果
    # source, target = 1088, 1118  # 西南财大 -> 骡马市 -> 火车南站 -> 神仙树
    source, target = 1092, 1117
    sta_dict = pd.read_pickle(os.path.join(DATA_DIR, "STA.pkl")) \
        .drop_duplicates(subset="STATION_UID") \
        .reset_index().set_index("STATION_UID")['STATION_NAME'].to_dict()
    print(sta_dict)
    print(sta_dict[source], " -> ", sta_dict[target])

    length, s_path = nx.single_source_dijkstra(net.G, source=source, target=target)
    print(length, s_path)
    # pass_info = net.get_passing_info(path)
    # print(pass_info)
    # pass_info_compact = net.compress_passing_info(pass_info)
    # print(pass_info_compact)
    # trans_cnt = net.get_trans_cnt(passing_info_compact=pass_info_compact)
    # print("Number of transfers: ", trans_cnt)

    max_length = min(length * 2, length + 1200)
    print(max_length)
    # paths = nx.all_simple_paths(net.G, source, target, cutoff=max_length)
    #
    # print(*paths)

    # net.find_all_pairs_k_paths()

    lens, paths = net.find_k_paths_via_yen(
        shortest_path=s_path, shortest_path_length=length, max_length=max_length, k=20
    )
    for le, pa in zip(lens, paths):
        print(le, net.compress_passing_info(path=pa))
        # print(pa)
    # print("K-paths", [net.compress_passing_info(net.get_passing_info(pa)) for pa in paths])

    pass


if __name__ == '__main__':
    main()
