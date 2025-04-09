# Prepare the network from Station and timetable datasets.
# This script will generate node_info.pkl, link_info.pkl, path.pkl, and pathvia.pkl.
import pandas as pd

from src.utils import execution_timer, file_saver, read_data
from src.metro_net import gen_links, gen_node_from_sta, ChengduMetro


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
    df_p, df_pv = net.find_all_pairs_k_paths()
    file_saver(lambda save_fn: df_p)(save_fn="path")
    file_saver(lambda save_fn: df_pv)(save_fn="pathvia")
    return


def main():
    nodes, links = get_node_and_link(read_=True)
    gen_path(nodes, links)
    return


if __name__ == "__main__":
    main()
    pass
