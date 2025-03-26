def main():
    import networkx as nx
    import numpy as np

    from src.metro_net import ChengduMetro
    from src.utils import read_data

    nodes = read_data("node_info", show_timer=False)
    links = read_data("link_info", show_timer=False)
    net = ChengduMetro(nodes, links)
    # net.plot_metro_net(coordinates=read_data("coordinates.csv"))

    source, target = np.random.choice(range(1001, 1137), 2)

    # cost, path = nx.single_source_dijkstra(net.G, source=source, target=target)

    # print(cost, path)

    print(list(nx.shortest_simple_paths(net.G, source, target, weight='weight')))


    # net._cal_path_perceived_cost(path)
    # net.find_k_shortest_paths()
    pass


if __name__ == '__main__':
    main()
