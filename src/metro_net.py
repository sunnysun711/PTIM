# build from TT, STA.
# 所有UID是都是地面层，nid分为上行和下行，表示站台，换乘需要从站台（nid）到达地面（uid）然后再去站台（nid），站台与地面层有一个极小的cost
import itertools
from typing import Iterable

import networkx as nx
import pandas as pd

from src.utils import read_data, read_platform_exceptions, file_saver


@file_saver
def gen_node_from_sta(save_fn: str = None) -> pd.DataFrame:
    """
    Generate node information from the station file.
    UID is used for ground surfaces, NID * 10 + 0 is for downward platform, NID * 10 + 1 is for upward platform.

    :param save_fn: file path to save node info.
    :return: Dataframe with columns ('STATION_NID', 'STATION_UID', 'IS_TRANSFER', 'IS_TERMINAL', 'LINE_NID') and
        index ('node_id').
    """
    df = read_data(fn="STA").reset_index()
    df1 = df.copy()
    df2 = df.copy()
    df1["node_id"] = df1["STATION_NID"] * 10
    df1["updown"] = 1  # downward
    df2["node_id"] = df2["STATION_NID"] * 10 + 1
    df2["updown"] = -1  # upward
    df["node_id"] = df['STATION_UID']
    df["updown"] = 0  # ground
    nodes = pd.concat(
        [
            df1,
            df2,
            df.drop_duplicates(subset=["node_id"]).drop(columns=["STATION_NID", "LINE_NID"])
        ],
        ignore_index=True
    )
    nodes['STATION_NID'] = nodes['STATION_NID'].astype("Int64")
    nodes['LINE_NID'] = nodes['LINE_NID'].astype("Int8")
    nodes['node_id'] = nodes['node_id'].astype("Int64")
    nodes['updown'] = nodes['updown'].astype("int8")

    assert nodes.node_id.unique().size == nodes.shape[0], "node id not unique."

    nodes.set_index("node_id", inplace=True)

    return nodes


def gen_train_links_from_tt() -> pd.DataFrame:
    """
    Generate train links from timetable.
    The link time is calculated with: Departure_ts - Stop_time / 2.
    Median values are used for each link.

    Note that:
                                    number of passing trains    类型
        STATION_NID STATION_NID.next
        10727       10722               1                             ^
        10413       10422               1                             ^
        10246       10252               1                             ^
        10154       10140               1                             ^
        10221       10246               1                             ^
        11026       11021               1                             ^
        10722       10727               1                             ^
        10141       10154               1                             ^
        10727       10741               1                             ^
                    10742               1                             ^
        10321       10337               1                             ^
        10729       10727               1                             ^
        10337       10321               1                             ^
        10120       10143               1                             ^
        11021       11026               1                             ^
        10741       10727               2                             ^
        10154       10141               5                             ^
        10252       10246              10                             ^ 出入段弧
        11025       11024             116                             * 正常区间
        11024       11023             116                             *
        11026       11025             116                             *
        11021       11022             116                             *
        11023       11022             116                             *
    So Number of passing trains is considered bigger than NO_MIN_PASS_TRAIN, defaults to 20.


    :return: Dataframe of indices (nid1, nid2, updown) and columns (count, time)
    """
    NO_MIN_PASS_TRAIN = 20

    df = read_data(fn="TT") \
        .drop(columns=["STATION_UID"]) \
        .sort_values(by=["TRAIN_ID", "ARRIVE_TS"]) \
        .reset_index()

    # use the mean times of (arrive_ts) and (departure_ts)
    df['ts1'] = df['DEPARTURE_TS'] - df['STOP_TIME'] / 2
    df['ts2'] = df['ts1'].shift(-1)
    df['STATION_NID2'] = df['STATION_NID'].shift(-1)
    df['TRAIN_ID2'] = df['TRAIN_ID'].shift(-1)

    # check link validity
    df = df.dropna(subset=["TRAIN_ID2"])
    df = df[df['TRAIN_ID'] == df['TRAIN_ID2']]

    df['time'] = df['ts2'] - df['ts1']

    gb = df.groupby(["STATION_NID", "STATION_NID2", "UPDOWN"])
    train_links = pd.DataFrame(index=gb.indices.keys()).rename_axis(index=['nid1', 'nid2', 'updown'])
    train_links["count"] = gb['time'].count()
    # delete special links (to and from train parks)
    train_links = train_links[train_links['count'] > NO_MIN_PASS_TRAIN]
    train_links['time'] = gb['time'].median()

    return train_links


def gen_walk_links_from_nodes(
        nodes: pd.DataFrame = None,
        platform_swap_time: float = 3,
        entry_time: float = 15,
        egress_time: float = 15,
        platform_exceptions: dict[int, list[list[int]]] = None,
) -> pd.DataFrame:
    """
    Generate walk links from nodes.
    :param nodes: Dataframe with columns ('STATION_NID', 'STATION_UID', 'IS_TRANSFER', 'IS_TERMINAL', 'LINE_NID') and
        index ('node_id'). Defaults to None, means reading from `read_data(fn='node_info')`.
    :param platform_swap_time: Defaults to 4 seconds.
    :param entry_time: Defaults to 15 seconds.
    :param egress_time: Defaults to 15 seconds.
    :param platform_exceptions: dict[uid, [[node_id, node_id], [node_id]]]. A dictionary where keys are station uids
        and values are lists of connected platform node ids. Represents special platform connection cases.
        Defaults to None, means generated from `read_platform_exceptions()`.
    :return: Dataframe of columns ['node_id1', 'node_id2', 'link_type', 'link_weight'].
    """
    platform_exceptions = read_platform_exceptions() if platform_exceptions is None else platform_exceptions
    # print(platform_exceptions)
    nodes = read_data(fn="node_info").dropna(subset=["LINE_NID"]) if nodes is None else nodes

    links = []  # [node_id1, node_id2, link_type, link_weight]

    # get entry, egress links
    for (ground_node,), platform_nodes_info in nodes[["STATION_UID"]].groupby(["STATION_UID"]):
        for ground_node_id, platform_node_id in itertools.product([ground_node], platform_nodes_info.index):
            links.append([ground_node_id, platform_node_id, "entry", entry_time])
            links.append([platform_node_id, ground_node_id, "egress", egress_time])

    # get platform swap links
    processed_special_uids = set()
    for (uid, line_nid), platform_nodes_info in nodes[["STATION_UID", "LINE_NID"]].groupby(
            by=["STATION_UID", "LINE_NID"]):
        if uid in platform_exceptions:
            if uid in processed_special_uids:
                continue
            processed_special_uids.add(uid)
            for platform_nodes in platform_exceptions[uid]:
                if len(platform_nodes) >= 2:
                    for plat1, plat2 in itertools.permutations(platform_nodes, 2):
                        links.append([plat1, plat2, "platform_swap", platform_swap_time])
        else:
            for plat1, plat2 in itertools.permutations(platform_nodes_info.index, 2):
                links.append([plat1, plat2, "platform_swap", platform_swap_time])

    # get dataframe
    walk_links = pd.DataFrame(links, columns=["node_id1", "node_id2", "link_type", "link_weight"], )
    walk_links = walk_links.astype({"node_id1": "int", "node_id2": "int", "link_type": "str", })

    # print(walk_links.sample(n=10))
    return walk_links


@file_saver
def gen_links(save_fn: str = None) -> pd.DataFrame:
    """

    :param save_fn:
    :return: Dataframe of columns ['node_id1', 'node_id2', 'link_type', 'link_weight'].
    """
    train_links = gen_train_links_from_tt().reset_index().drop(columns=["count"])
    train_links['node_id1'] = (train_links['nid1'] * 10).astype(int)
    train_links['node_id2'] = (train_links['nid2'] * 10).astype(int)
    train_links['link_type'] = "in_vehicle"
    train_links.loc[train_links['updown'] == -1, "node_id1"] += 1
    train_links.loc[train_links['updown'] == -1, "node_id2"] += 1
    train_links.drop(columns=["nid1", "nid2", "updown"], inplace=True)
    train_links.rename(columns={"time": "link_weight"}, inplace=True)

    walk_links = gen_walk_links_from_nodes()

    all_links = pd.concat([train_links, walk_links], ignore_index=True)

    return all_links


def find_shortest_path(G: nx.DiGraph, source_id: int, target_id: int) -> tuple[float, list[int]]:
    """
    :return: tuple[path cost, list[passing nodes]]
    """
    return nx.single_source_dijkstra(G, source_id, target_id)


class ChengduMetro:
    def __init__(self, nodes: pd.DataFrame, links: pd.DataFrame):
        self.nodes = nodes  # 'node_id' (index), 'STATION_NID', 'STATION_UID', 'IS_TRANSFER', 'IS_TERMINAL', 'LINE_NID'

        # add passing line
        links['passing_line'] = 0  # entry, egress, platform_swap: 0
        links['updown'] = 0  # entry, egress, platform_swap: 0
        links.loc[
            links['link_type'] == "in_vehicle", "passing_line"
        ] = (
                    links.loc[links['link_type'] == "in_vehicle", "node_id1"] // 10 - 10000
            ) // 100

        links.loc[
            (links['link_type'] == "in_vehicle") & (links["node_id1"] % 10 == 0), "updown"
        ] = 1  # downward
        links.loc[
            (links['link_type'] == "in_vehicle") & (links["node_id1"] % 10 == 1), "updown"
        ] = -1  # upward

        self.links = links  # 'node_id1', 'node_id2', 'link_type', 'link_weight', 'passing_line', 'updown'
        self.G = nx.DiGraph()

        node_info_list = [(k, v) for k, v in nodes.to_dict("index").items()]
        self.G.add_nodes_from(node_info_list)

        link_info_list = [
            (
                row.node_id1, row.node_id2,
                {"type": row.link_type, "weight": row.link_weight,
                 "line": row.passing_line, "updown": row.updown}
            )
            for row in links.itertuples()
        ]
        self.G.add_edges_from(link_info_list)

    def _get_nid_uid_dict(self) -> dict[int, int]:
        _df = self.nodes[["STATION_NID", "STATION_UID"]].drop_duplicates().set_index("STATION_NID")
        return _df["STATION_UID"].to_dict()

    def _get_uid_list(self) -> Iterable[int]:
        # uids = sorted(self.nodes['STATION_UID'].to_list())
        # return set(uids)
        return range(1001, 1137)

    def print_graph_info(self):
        # 打印图的信息来验证
        print(f"Total nodes: {len(self.G.nodes)}")
        print(f"Total edges: {len(self.G.edges)}")
        print("Nodes with attributes:")
        for node in self.G.nodes(data=True):
            print(node)
        print("Edges with attributes:")
        for edge in self.G.edges(data=True):
            print(edge)

    def plot_metro_net(self, coordinates: pd.DataFrame):
        """A simple networkx based plotting function for metro network."""
        uid_dict = self._get_nid_uid_dict()

        coordinates = coordinates.set_index("station_nid")
        coordinates['uid'] = uid_dict
        coordinates.drop_duplicates(subset="uid", inplace=True)
        node_info_list = [(row.uid, {"pos": (row.x, row.y)}) for row in coordinates.itertuples()]

        links = self.links[self.links["link_type"] == "in_vehicle"].drop(columns=["link_type"]).copy()
        links["node_id1"], links["node_id2"] = links["node_id1"] // 10, links["node_id2"] // 10
        link_info_list = [
            (uid_dict[row.node_id1], uid_dict[row.node_id2], {"weight": row.link_weight})
            for row in links.itertuples()
        ]
        # generate a graph object
        G = nx.DiGraph()
        G.add_nodes_from(node_info_list)
        G.add_edges_from(link_info_list)

        options = {
            "font_size": 8,
            "node_size": 500,
            "node_color": "white",
            "edgecolors": "gray",
            "linewidths": 1,
            "width": 2,
            "with_labels": True,
        }

        pos = nx.get_node_attributes(G, 'pos')

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, **options)
        plt.show()
        return

    def plot_metro_net_with_plotly(self, coordinates: pd.DataFrame, _paths: list[list[int]]):
        """
        A plotly based 3D plotting function for metro network. Optionally displaying paths with different colors.

        Node_id, instead of STATION_UID, is used for plotting.

        每条线路都有一个固定且互不相同的高度坐标，in-vehicle link可以用黑色表示，entry 和 egress links 可以用深灰色表示，
        platform swap link 可以用淡灰色表示。 k短路各自用不同的颜色，叠加在上面表示，可以选择性开启或关闭。

        （后期再实现这个功能）
        """
        ...

    def get_passing_info(self, path: list[int]) -> list[tuple[str, int, int]]:
        """
        Get a list of link types, passing lines, and updown directions for a path.
        :param path: list of node_id.
        :return: list of linke types, passing lines and updown directions.
            For e.g. [(type1, line1, upd1), (type2, line2, upd2), ...].
        """
        passing_info = []
        for i, j in zip(path[:-1], path[1:]):
            cur_edge = self.G.edges[i, j]
            cur_info = (cur_edge["type"], cur_edge["line"], cur_edge["updown"])
            if len(passing_info) == 0 or passing_info[-1] != cur_info:
                passing_info.append(cur_info)
        return passing_info

    def compress_passing_info(self, passing_info: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
        lines_upd = [(i[1], i[2]) for i in passing_info if i[1] != 0]
        return lines_upd

    def get_trans_cnt(self, path: list[int] = None, passing_info: list[tuple[str, int, int]] = None,
                      passing_info_compact: list[tuple[int, int]] = None) -> int:
        """Get transfer count from either path, passing_info or passing_info_compact."""
        if passing_info_compact is not None:
            lines_upd = passing_info_compact
        elif passing_info is not None:
            lines_upd = self.compress_passing_info(passing_info)
        elif path is not None:
            passing_info = self.get_passing_info(path)
            lines_upd = self.compress_passing_info(passing_info)
        else:
            raise ValueError("At least one of path, passing_info, or passing_info_compact must be provided.")
        return len(lines_upd) - 1

    def find_k_shortest_paths(self, k: int = 8, theta1: float = 0.6, theta2: float = 600, ):

        # 思路：
        uid_list = self._get_uid_list()  # all ground nodes
        for o_uid in uid_list:
            # get all nodes shortest paths with source as o_uid
            lengths, s_paths = nx.single_source_dijkstra(G=self.G, source=o_uid)
            for d_uid in uid_list:
                if o_uid == d_uid:
                    continue
                shortest_path = s_paths[d_uid]
                shortest_path_length = lengths[d_uid]

                max_path_length = min(shortest_path_length * (1 + theta1), shortest_path_length + theta2)

                print(shortest_path, shortest_path_length, max_path_length)

                # ！不可以用 <nx.all_simple_paths> 和 max_path_length 作为 cutoff 来生成k短路，计算太慢。
                # Ref: https://blog.csdn.net/sheyueyu/article/details/133774669
                # 在我的方法中，只移除 in_vehicle 弧，进出站、换站台都【不可】被移除。


        ...
