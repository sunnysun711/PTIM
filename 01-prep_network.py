from src.metro_net import *


def main():
    # generate node and link files
    nodes = gen_node_from_sta(save_fn="node_info")
    links = gen_links(save_fn="link_info")

    # generate k-paths files
    net = ChengduMetro(nodes=nodes, links=links)
    net.find_all_pairs_k_paths(k=8)


if __name__ == "__main__":
    main()
    pass
