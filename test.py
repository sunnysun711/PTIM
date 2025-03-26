def main():
    import networkx as nx
    import numpy as np

    from src.metro_net import ChengduMetro
    from src.utils import read_data

    nodes = read_data("node_info", show_timer=False)
    links = read_data("link_info", show_timer=False)
    net = ChengduMetro(nodes, links)

    # net.plot_metro_net(coordinates=read_data("coordinates.csv"))

    # source, target = np.random.choice(range(1001, 1137), 2)
    # cost, path = nx.single_source_dijkstra(net.G, source=source, target=target)
    # print(cost, path)
    # pass_info = net.get_passing_info(path)
    # print(pass_info)
    # pass_info_compact = net.compress_passing_info(pass_info)
    # print(pass_info_compact)
    # trans_cnt = net.get_trans_cnt(passing_info_compact=pass_info_compact)
    # print("Number of tranasfers: ", trans_cnt)

    net.find_k_shortest_paths()

    pass


if __name__ == '__main__':
    main()
