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

from src.utils import execution_timer, file_saver, read_data
from src.metro_net import gen_links, gen_node_from_sta, ChengduMetro, nx


def _test_k_paths():
    """Test the find_k_paths_via_yen function. Check if `_check_path_feas` works."""
    node, link = read_data("node_info", show_timer=False), read_data("link_info", show_timer=False)
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


def get_node_and_link(read_: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not read_:
        # generate node and link files
        nodes = gen_node_from_sta(save_fn="node_info")
        links = gen_links(save_fn="link_info")
    else:
        nodes = read_data("node_info")
        links = read_data("link_info")
    return nodes, links


@execution_timer
def gen_path(nodes: pd.DataFrame, links: pd.DataFrame):
    # generate k-paths files
    net = ChengduMetro(nodes=nodes, links=links)
    df_p, df_pv = net.find_all_pairs_k_paths()  # takes 3 hours to run
    file_saver(lambda save_fn: df_p)(save_fn="path")
    file_saver(lambda save_fn: df_pv)(save_fn="pathvia")
    return


def main():
    print("\033[33m"
          "======================================================================================"
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
    nodes, links = get_node_and_link(read_=False)
    print("\033[33m"
          "[INFO] Generating path and pathvia files...\n"
          "\033[0m")
    gen_path(nodes, links)
    return


if __name__ == "__main__":
    # _test_k_paths()
    # main()
    pass
