"""
This script prepares the metro network structure for pathfinding.
It generates nodes, links, k-shortest paths, and their segment breakdowns.

Key Outputs:
- node_info.pkl: Metro station and platform node information
- link_info.pkl: Travel and walking links between nodes
- path.pkl: K-shortest paths with metadata
- pathvia.pkl: Path segment details for each OD path

Dependencies:
- src.metro_net
- src.utils
"""
import pandas as pd

from src import config
from src.utils import execution_timer, read_, save_
from src.metro_net import gen_links, gen_node_from_sta, ChengduMetro, nx


def _test_k_paths():
    """Test the find_k_paths_via_yen function. Check if `_check_path_feas` works."""
    node = read_(config.CONFIG["results"]["node"], show_timer=False)
    link = read_(config.CONFIG["results"]["link"], show_timer=False)
    net = ChengduMetro(nodes=node, links=link)

    # net.find_all_pairs_k_paths()

    source, target = 1136, 1097

    s_lengths, s_paths = nx.single_source_dijkstra(G=net.G, source=source)
    s_path, s_path_length = s_paths[target], s_lengths[target]
    max_length = min(s_path_length * (1 + 0.6), s_path_length + 600)
    k_lens, k_paths = net.find_k_paths_via_yen(shortest_path=s_path, shortest_path_length=net.cal_path_length(s_path),
                                               max_length=max_length)
    print(k_lens)
    for path in k_paths:
        print(net.compress_passing_info(path=path))

    return


def get_node_and_link(read_on: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not read_on:
        # generate node and link files
        node = gen_node_from_sta(save_on=True)
        link = gen_links(save_on=True)
    else:
        node = read_(config.CONFIG["results"]["node"], show_timer=False)
        link = read_(config.CONFIG["results"]["link"], show_timer=False)
    return node, link


@execution_timer
def gen_path(nodes: pd.DataFrame, links: pd.DataFrame):
    # generate k-paths files
    net = ChengduMetro(nodes=nodes, links=links)
    df_p, df_pv = net.find_all_pairs_k_paths(
        k=config.CONFIG["parameters"]["k"],
        theta1=config.CONFIG["parameters"]["theta1"],
        theta2=config.CONFIG["parameters"]["theta2"],
        transfer_deviation=config.CONFIG["parameters"]["transfer_deviation"]
    )  # takes 3 hours to run
    save_(config.CONFIG["results"]["path"], df_p, auto_index_on=False)
    save_(config.CONFIG["results"]["pathvia"], df_pv, auto_index_on=False)
    return


def main(read_network: bool):
    print("\033[33m"
          "======================================================================================\n"
          "[INFO] This script prepares the metro network structure for pathfinding.\n"
          "       It generates nodes, links, k-shortest paths, and their segment breakdowns.\n"
          "       Key Outputs:\n"
          "       - node_info.pkl: Metro station and platform node information\n"
          "       - link_info.pkl: Travel and walking links between nodes\n"
          "       - path.pkl: K-shortest paths with metadata\n"
          "       - pathvia.pkl: Path segment details for each OD path\n"
          "======================================================================================"
          "\033[0m")
    print("\033[33m"
          "[INFO] Generating node and link files...\n"
          "\033[0m")
    nodes, links = get_node_and_link(read_on=read_network)
    print("\033[33m"
          "[INFO] Generating path and pathvia files...\n"
          "\033[0m")
    gen_path(nodes, links)
    return


if __name__ == "__main__":
    # _test_k_paths()
    # main()
    config.load_config()
    nodes, links = get_node_and_link(read_on=True)
    net = ChengduMetro(nodes=nodes, links=links)
    import time
    a = time.time()
    df_p, df_pv = net.find_all_pairs_k_paths_parallel()
    df_p.info()
    df_pv.info()
    b = time.time()
    print(f"Elapsed time: {b - a} seconds")
    pass
