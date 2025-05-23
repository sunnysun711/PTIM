"""
This module constructs the metro network graph using station and timetable data.
It builds a directed graph for shortest-path and k-path calculations.

Key Classes and Functions:
1. gen_node_from_sta: Constructs node representation from station data
2. gen_train_links_from_tt: Generates links between platforms based on train service
3. gen_walk_links_from_nodes: Generates walking connections for entry, egress, and platform swaps
4. ChengduMetro: Core graph class with pathfinding and visualization utilities
5. ChengduMetro.find_all_pairs_k_paths: Generates k-shortest paths for all OD pairs

Dependencies:
- pandas, numpy: For data handling
- networkx: For graph representation
- src.utils: For reading and saving data

Data Sources:
- STA.pkl, TT.pkl, platform.json
"""
import itertools
from heapq import heappush, heappop
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src import config
from src.globals import get_platform_exceptions, get_node_info
from src.utils import read_, save_


def gen_node_from_sta(save_on: bool = False) -> pd.DataFrame:
    """
    Generate node information from the station file.

    UID is used for ground surfaces, NID * 10 + 0 is for downward platform, NID * 10 + 1 is for upward platform.

    :param save_on: Whether to save the result to a file. Default is False.
    :type save_on: bool
    
    :return: Dataframe with columns ('STATION_NID', 'STATION_UID', 'IS_TRANSFER', 'IS_TERMINAL', 'LINE_NID', 'updown') and
        index ('node_id').
    :rtype: pd.DataFrame
    """
    df = read_(fn="STA", show_timer=False).reset_index()
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

    if save_on:
        save_(fn=config.CONFIG["results"]["node"], data=nodes, auto_index_on=False)

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
    :rtype: pd.DataFrame
    """
    NO_MIN_PASS_TRAIN = 20

    df = read_(fn="TT", show_timer=False) \
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
        index ('node_id'). 
        
        Defaults to None, means reading from `read_(fn='node_info')`.
    :type nodes: pd.DataFrame
    :param platform_swap_time: Defaults to 4 seconds.
    :type platform_swap_time: float
    :param entry_time: Defaults to 15 seconds.
    :type entry_time: float
    :param egress_time: Defaults to 15 seconds.
    :type egress_time: float
    :param platform_exceptions: dict[uid, [[node_id, node_id], [node_id]]]. 
        A dictionary where keys are station uids
        and values are lists of connected platform node ids. 
        
        Represents special platform connection cases.
        
        Defaults to None, means generated from `read_("platform.json")`.
    :type platform_exceptions: dict[int, list[list[int]]]
    :return: Dataframe of columns ['node_id1', 'node_id2', 'link_type', 'link_weight'].
    :rtype: pd.DataFrame
    """
    platform_exceptions = get_platform_exceptions() if platform_exceptions is None else platform_exceptions
    # print(platform_exceptions)
    nodes = get_node_info().dropna(subset=["LINE_NID"]).set_index("node_id") if nodes is None else nodes

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


def gen_links(platform_swap_time: float = 3,
              entry_time: float = 15,
              egress_time: float = 15,
              save_on: bool = False) -> pd.DataFrame:
    """
    Generate the links (train and walk) between metro nodes.

    This function combines the train links generated from the timetable (`gen_train_links_from_tt`) and walk links
    between nodes (`gen_walk_links_from_nodes`). It creates a unified DataFrame that contains information about all
    the possible links in the metro network, including train (in-vehicle) and walk (entry, egress, platform swap) links.

    - Train links are created based on the timetable and include information like the time taken for a train to travel
      between two stations.
    - Walk links are generated based on the nodes' relationships, representing entry, egress, and platform swap times
      between platforms or platforms and station gates.

    :param platform_swap_time: The time taken to swap platforms, default is 3 seconds.
    :type platform_swap_time: float
    :param entry_time: The time taken to walk from station gates to the platforms, default is 15 seconds.
    :type entry_time: float
    :param egress_time: The time taken to walk from a platform to the station gates, default is 15 seconds.
    :type egress_time: float
    :param save_on: Whether to save the generated links to a file, default is False.
    :type save_on: bool
    :return: Dataframe of columns ['node_id1', 'node_id2', 'link_type', 'link_weight'].
    :rtype: pd.DataFrame
    """
    train_links = gen_train_links_from_tt().reset_index().drop(columns=["count"])
    train_links['node_id1'] = (train_links['nid1'] * 10).astype(int)
    train_links['node_id2'] = (train_links['nid2'] * 10).astype(int)
    train_links['link_type'] = "in_vehicle"
    train_links.loc[train_links['updown'] == -1, "node_id1"] += 1
    train_links.loc[train_links['updown'] == -1, "node_id2"] += 1
    train_links.drop(columns=["nid1", "nid2", "updown"], inplace=True)
    train_links.rename(columns={"time": "link_weight"}, inplace=True)

    walk_links = gen_walk_links_from_nodes(
        platform_swap_time=platform_swap_time, entry_time=entry_time, egress_time=egress_time
    )

    all_links = pd.concat([train_links, walk_links], ignore_index=True)

    if save_on:
        save_(fn=config.CONFIG["results"]["link"], data=all_links, auto_index_on=False)
    return all_links


def gen_platforms() -> pd.DataFrame:
    """
    Generate physical platforms from the exceptions provided in platform.json and the default rules
    that node_ids sharing the same LINE_NID use the same physical platform.
        
    The physical platform id is generated as id + uid * 100, where uid is the station uid.
    
    :return: Dataframe with columns ['physical_platform_id', 'node_id', 'uid'].
    :rtype: pd.DataFrame
    """
    platform_excep = get_platform_exceptions()
    nodes = get_node_info().dropna(subset=["LINE_NID"])[["node_id", "STATION_UID", "LINE_NID"]].set_index("node_id")  # excluding ground nodes
    
    data = []  # [uid, physical_platform_id, node_id] 
    for uid in range(1001, 1137):
        physical_platform_id = 1 + uid * 100
        if uid in platform_excep:
            platforms = platform_excep[uid]
            for platform in platforms:
                for node_id in platform:
                    data.append([physical_platform_id, node_id, uid])
                physical_platform_id += 1
        else:
            lines = nodes[nodes["STATION_UID"] == uid]["LINE_NID"].unique()
            for line in lines:
                node_ids = nodes[(nodes["STATION_UID"] == uid) & (nodes["LINE_NID"] == line)].index
                for node_id in node_ids:
                    data.append([physical_platform_id, node_id, uid])
                physical_platform_id += 1
                
    res = pd.DataFrame(data, columns=["physical_platform_id", "node_id", "uid"])
    return res


class ChengduMetro:
    def __init__(self, nodes: pd.DataFrame, links: pd.DataFrame):
        """
        Initialize the ChengduMetro object with nodes and links data.

        :param nodes: Dataframe of columns ['node_id' (index), 'STATION_NID', 'STATION_UID', 'IS_TRANSFER',
            'IS_TERMINAL', 'LINE_NID']
        :type nodes: pd.DataFrame
        :param links: Dataframe of columns ['node_id1', 'node_id2', 'link_type', 'link_weight']
        :type links: pd.DataFrame
        """
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
        self.nid2uid = self._get_nid_uid_dict()

    def _get_nid_uid_dict(self) -> dict[int, int]:
        """
        Get a dictionary mapping NID to UID.

        :return: A dictionary with NID as keys and UID as values.
        :rtype: dict[int, int]
        """
        _df = self.nodes[["STATION_NID", "STATION_UID"]].drop_duplicates().set_index("STATION_NID")
        return _df["STATION_UID"].to_dict()

    def _get_uids(self) -> Iterable[int]:
        """
        Get a list of unique UIDs.

        :return: An iterable of unique UIDs.
        :rtype: Iterable[int]
        """
        # uids = sorted(self.nodes['STATION_UID'].to_list())
        # return set(uids)
        return range(1001, 1137)

    def print_graph_info(self):
        print(f"Total nodes: {len(self.G.nodes)}")
        print(f"Total edges: {len(self.G.edges)}")
        print("Nodes with attributes:")
        for node in self.G.nodes(data=True):
            print(node)
        print("Edges with attributes:")
        for edge in self.G.edges(data=True):
            print(edge)

    def plot_metro_net(self, coordinates: pd.DataFrame):
        """
        A simple networkx based plotting function for metro network.

        :param coordinates: Dataframe containing node coordinates. columns ["station_nid", "x", "y"]
        :type coordinates: pd.DataFrame
        """
        coordinates = coordinates.set_index("station_nid")
        coordinates['uid'] = self.nid2uid
        coordinates.drop_duplicates(subset="uid", inplace=True)
        node_info_list = [(row.uid, {"pos": (row.x, row.y)}) for row in coordinates.itertuples()]

        links = self.links[self.links["link_type"] == "in_vehicle"].drop(columns=["link_type"]).copy()
        links["node_id1"], links["node_id2"] = links["node_id1"] // 10, links["node_id2"] // 10
        link_info_list = [
            (self.nid2uid[row.node_id1], self.nid2uid[row.node_id2], {"weight": row.link_weight})
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

    def cal_path_length(self, path: list[int]) -> float:
        cost = 0
        for i, j in zip(path[:-1], path[1:]):
            cost += self.G.edges[i, j]["weight"]
        return cost

    def get_passing_uid(self, path: list[int], merge_: bool = False) -> list[int]:
        """Get passing uid list from path, same consecutive uids are merged as one."""
        uid_path = [self.nid2uid[node_id // 10] if node_id > 1e5 else node_id for node_id in path]

        if merge_:  # merge consecutive uid
            uid_path_ = [uid_path[0]]
            for uid in uid_path[1:]:
                if uid != uid_path_[-1]:
                    uid_path_.append(uid)
            return uid_path_
        return uid_path

    def get_passing_info(self, path: list[int]) -> list[tuple[str, int, int]]:
        """
        Get a list of link types, passing lines, and updown directions for a path.

        Consecutive links of the same lines are not merged here. Use compress_passing_info for merged results.

        :param path: list of node_id.
        :type path: list[int]
        :return: list of linke types, passing lines and updown directions.
            
            For e.g. [(type1, line1, upd1), (type2, line2, upd2), ...].
        :rtype: list[tuple[str, int, int]]
        """
        passing_info = []
        for i, j in zip(path[:-1], path[1:]):
            cur_edge = self.G.edges[i, j]
            cur_info = (cur_edge["type"], cur_edge["line"], cur_edge["updown"])
            # if len(passing_info) == 0 or passing_info[-1] != cur_info:
            passing_info.append(cur_info)
        return passing_info

    def get_path_via(self, path: list[int], path_id: int) -> list[list]:
        """
        Generate path via information for a given path in the metro network.

        This function processes a given path (list of node IDs) and generates a list of segments along the path,
        where each segment is described by the link type (e.g., entry, egress, platform_swap, in_vehicle), the passing
        line, and the direction (up/down). The path is divided into sections based on the type of link, and each section
        is assigned a unique pathvia ID (`pv_id`) for sorting purposes.

        The function creates a list of passing information for the given path by iterating through consecutive nodes
        and adding the appropriate information for each link between them.

        :param path: A list of node IDs representing the path through the metro network.
        :type path: list[int]
        :param path_id: A unique identifier for the path, used for sorting and distinguishing between different paths.
        :type path_id: int

        :return: A list of lists, where each inner list represents a segment of the path and contains the following
            information:

            - `path_id`: The unique ID for the path.
            - `pv_id`: The pathvia ID, used for sorting the segments.
            - `node_id1`: The starting node ID of the segment.
            - `node_id2`: The ending node ID of the segment.
            - `link_type`: The type of the link (e.g., "entry", "egress", "platform_swap", "in_vehicle").
            - `line_nid`: The line number associated with the segment.
            - `updown`: The direction of the link (1 for upward, -1 for downward).
        """
        passing_info_arr = []
        cur_line_first_sec = None
        cur_line_first_node = 0
        pv_id = 1  # pathvia id for sorting the segments
        for i, j in zip(path[:-1], path[1:]):
            cur_edge = self.G.edges[i, j]
            if cur_edge["type"] != "in_vehicle":
                if cur_line_first_sec is not None:
                    passing_info_arr.append(
                        [path_id, pv_id, cur_line_first_node, i, cur_line_first_sec["type"], cur_line_first_sec["line"],
                         cur_line_first_sec["updown"]])
                    pv_id += 1
                    cur_line_first_sec = None
                passing_info_arr.append([path_id, pv_id, i, j, cur_edge["type"], cur_edge["line"], cur_edge["updown"]])
                pv_id += 1
            else:
                if cur_line_first_sec is None:
                    cur_line_first_sec = cur_edge
                    cur_line_first_node = i

        return passing_info_arr

    def compress_passing_info(
            self,
            passing_info: list[tuple[str, int, int]] = None,
            path: list[int] = None) -> list[str]:
        """
        Compress the passing information into a more compact form.

        :param passing_info: List of tuples containing link types, passing lines, and updown directions.
        :type passing_info: list[tuple[str, int, int]] (optional)
        :param path: List of node id. Default to None, if both provided, use passing_info first.
        :type path: list[int] (optional)
        :return: Compressed list of str. For e.g. ['line1|upd1|number_of_sections1', ...].
        """
        if passing_info is None:
            passing_info = self.get_passing_info(path)
        cur_line_sections = 0
        last_line, last_upd = None, None
        lines_upd = []

        for tp, li, upd in passing_info:
            if tp != "in_vehicle":
                continue
            if last_line is None or (li == last_line and upd == last_upd):
                cur_line_sections += 1
            else:
                lines_upd.append(f"{last_line}|{last_upd}|{cur_line_sections}")
                cur_line_sections = 1
            last_line, last_upd = li, upd
        else:
            lines_upd.append(f"{last_line}|{last_upd}|{cur_line_sections}")
        return lines_upd

    def get_trans_cnt(self,
                      path: list[int] = None,
                      passing_info: list[tuple[str, int, int]] = None,
                      passing_info_compact: list[str] = None) -> int:
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

    def find_k_paths_via_yen(
            self,
            shortest_path: list[int],
            shortest_path_length: float,
            max_length: float,
    ) -> tuple[list[float], list[list[int]]]:
        """
        Find the k shortest paths for this OD pair with `shortest_path` as input.
        Cut off at either `max_length` or `k`.

        :param shortest_path: List of node_id for the shortest path.
        :type shortest_path: list[int]
        :param shortest_path_length: Length of the shortest path.
        :type shortest_path_length: float
        :param max_length: Maximum length of the k shortest path.
        :type max_length: float
        :return: List of tuples containing shortest path lengths and shortest paths.
        :rtype: tuple[list[float], list[list[int]]]
        """
        source, target = shortest_path[0], shortest_path[-1]
        lengths, paths = [shortest_path_length], [shortest_path]
        c = itertools.count()
        B = []  # all K-paths (topologically feasible)
        G = self.G.copy()

        while True:
            for j in range(len(paths[-1]) - 1):
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]
                root_path_length = self.cal_path_length(root_path)

                edges_removed = []
                # remove nodes connected to the spur_node in current k-paths
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[: j + 1]:
                        to_remove_node = c_path[j + 1]
                        for u, v, edge_attr in [*G.in_edges(nbunch=to_remove_node, data=True),
                                                *G.out_edges(nbunch=to_remove_node, data=True)]:
                            edges_removed.append((u, v, edge_attr))
                # remove all edges connecting nodes in the root path
                to_remove_nodes = [root_path[n] for n in range(len(root_path) - 1)]
                for (u, v, edge_attr) in [*G.in_edges(nbunch=to_remove_nodes, data=True),
                                          *G.out_edges(nbunch=to_remove_nodes, data=True)]:
                    edges_removed.append((u, v, edge_attr))
                G.remove_edges_from(edges_removed)

                # find paths from spur_node
                spur_paths_length, spur_paths = nx.single_source_dijkstra(
                    G, spur_node, cutoff=max_length - root_path_length)

                # if found target within cutoff
                if target in spur_paths and spur_paths[target]:
                    # calculate path length
                    total_path = root_path[:-1] + spur_paths[target]
                    total_path_length = root_path_length + spur_paths_length[target]
                    heappush(B, (total_path_length, next(c), total_path))

                G.add_edges_from(edges_removed)

            if B:
                (l, _, p) = heappop(B)
                lengths.append(l)
                paths.append(p)
                # print(self.get_passing_info(p))
                # print("New path found: ", [self.cal_path_length(pa) for pa in paths])
            else:
                break

        # check uniqueness
        paths_, lengths_ = [], []
        for i, (p, p_len) in enumerate(zip(paths, lengths)):
            if p not in paths_:
                paths_.append(p)
                lengths_.append(p_len)
        # sort paths with lengths ascending
        sorted_index = np.argsort(lengths_)
        paths = [paths_[_] for _ in sorted_index]
        lengths = [lengths_[_] for _ in sorted_index]

        # check feasibility (practical) and save top k
        res_paths, res_lengths = [], []
        for p, p_len in zip(paths, lengths):
            if self._check_path_feas(path=p):
                res_paths.append(p)
                res_lengths.append(p_len)

        return res_lengths, res_paths

    def _check_path_feas(self, path: list[int]) -> bool:
        """check path feasibility"""
        # 1. detour check
        uid_list = self.get_passing_uid(path=path, merge_=True)
        for i, uid in enumerate(uid_list):
            if uid in uid_list[i + 1:] and uid not in [1043, 1098]:
                # detour found. special care to Line 1 SiHe and Huafu Avenue.
                return False

        # 2. walk detour check (platform_swap should only exist between in_vehicle links)
        passing_info = self.get_passing_info(path)
        for li1, li2 in zip(passing_info[:-1], passing_info[1:]):
            two_types = [li1[0], li2[0]]
            if "platform_swap" in two_types and ("entry" in two_types or "egress" in two_types):
                return False

        # 4. walk detour check 2 (transfer via egress-entry while platform_swap is available, check 1136 -> 1097)
        for nd1, nd2, nd3 in zip(path[:-2], path[1:-1], path[2:]):
            if self.G.has_edge(nd1, nd3) and self.G.edges[nd1, nd3]["type"] == "platform_swap":
                if self.G.has_edge(nd1, nd2) and self.G.has_edge(nd2, nd3) and \
                        self.G.edges[nd1, nd2]["type"] == "egress" and self.G.edges[nd2, nd3]["type"] == "entry":
                    return False

        # 3. change back to previous line_upd (except for line 7)
        line_upd_secs = self.compress_passing_info(passing_info=passing_info)
        line_upds = [(txt.split("|")[0], txt.split("|")[1]) for txt in line_upd_secs]
        for i, (li, upd) in enumerate(line_upds):
            if li != "7" and (li, upd) in line_upds[i + 1:]:  # change back to the same line_upd (except for line 7)
                return False

        return True

    def find_all_pairs_k_paths(
            self,
            k: int = 10,
            theta1: float = 0.6,
            theta2: float = 600,
            transfer_deviation: int = 2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        LEGACY METHOD: For better performance, please use find_all_pairs_k_paths_parallel.

        Find the k shortest paths for all OD pairs (only ground nodes).

        :param k: Number of shortest paths to find. Default to 10.
        :param theta1: Parameter for path selection. Default to 0.6.
        :param theta2: Parameter for path selection. Default to 600.
        :param transfer_deviation: Transfer deviation from the trans_cnt of shortest_path. Default to 2.
        :return: Path and pathvia dataframes.
            Path dataframe columns: ["path_id", "length", "transfer_cnt", "path_str"]
            Pathvia dataframe columns: ["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown"]
        """
        path_list: list[list] = []
        pathvia_list: list[list] = []

        uid_list = self._get_uids()  # all ground nodes
        for o_uid in uid_list:
            # get all nodes shortest paths with source as o_uid
            s_lengths, s_paths = nx.single_source_dijkstra(G=self.G, source=o_uid)
            for d_uid in tqdm(uid_list, desc=f"Finding K paths for origin: {o_uid}", unit=" OD Pair"):
                if o_uid == d_uid:
                    continue

                shortest_path = s_paths[d_uid]
                shortest_path_length = s_lengths[d_uid]
                max_path_length = min(shortest_path_length * (1 + theta1), shortest_path_length + theta2)
                lengths, paths = self.find_k_paths_via_yen(
                    shortest_path=shortest_path, shortest_path_length=shortest_path_length,
                    max_length=max_path_length
                )

                # check transfer count and select top k paths
                transfers = [self.get_trans_cnt(pa) for pa in paths]
                max_transfers = min(transfers) + transfer_deviation
                k_ = 0
                base_path_id = int(f"{o_uid}{d_uid}00")
                for transfer, length, path in zip(transfers, lengths, paths):
                    if transfer <= max_transfers:
                        k_ += 1
                        path_id = base_path_id + k_
                        path_list.append(
                            [path_id, length, transfer, "_".join(self.compress_passing_info(path=path))]
                        )
                        pv = self.get_path_via(path=path, path_id=path_id)
                        pathvia_list.extend(pv)
                    if k_ >= k:
                        break

        df_p = pd.DataFrame(path_list, columns=["path_id", "length", "transfer_cnt", "path_str"])
        df_pv = pd.DataFrame(
            pathvia_list, columns=["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown"])
        return df_p, df_pv

    def find_all_pairs_k_paths_parallel(
            self,
            k: int = 10,
            theta1: float = 0.6,
            theta2: float = 600,
            transfer_deviation: int = 2,
            n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parallel version: Find the k shortest paths for all OD pairs (only ground nodes).
        ETA: ~ 850 seconds (2025-04-20 PC i9-13900K)

        :param k: Number of shortest paths to find. Default to 10.
        :type k: int (optional)
        :param theta1: Relative tolerance for max path length.
        :type theta1: float (optional)
        :param theta2: Absolute tolerance for max path length.
        :type theta2: float (optional)
        :param transfer_deviation: Max allowed deviation in transfer count.
        :type transfer_deviation: int (optional)
        :param n_jobs: Number of parallel workers. -1 means using all processors.
        :type n_jobs: int (optional)
        :return: Path and pathvia dataframes.
            
            - Path dataframe columns: ["path_id", "length", "transfer_cnt", "path_str"]
            - Pathvia dataframe columns: ["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown"]
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        uid_list = self._get_uids()  # all ground nodes
        print(f"[INFO] Start finding K-shortest paths using {n_jobs} threads...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_origin)(
                o_uid=o_uid,
                uid_list=uid_list,
                k=k,
                theta1=theta1,
                theta2=theta2,
                transfer_deviation=transfer_deviation
            )
            for o_uid in tqdm(uid_list, desc="Parallel OD Path Search")
        )

        path_list_all = []
        pathvia_list_all = []
        for path_list, pathvia_list in results:
            path_list_all.extend(path_list)
            pathvia_list_all.extend(pathvia_list)

        df_p = pd.DataFrame(path_list_all, columns=["path_id", "length", "transfer_cnt", "path_str"])
        df_pv = pd.DataFrame(
            pathvia_list_all, columns=["path_id", "pv_id", "node_id1", "node_id2", "link_type", "line", "updown"]
        )
        return df_p, df_pv

    def _process_origin(
            self,
            o_uid: int,
            uid_list: list[int],
            k: int,
            theta1: float,
            theta2: float,
            transfer_deviation: int
    ) -> tuple[list[list], list[list]]:
        """
        Process one origin UID, computing k-shortest paths to all other destinations.
        Returns two lists: path summary and pathvia detail.
        """
        path_list = []
        pathvia_list = []

        try:
            s_lengths, s_paths = nx.single_source_dijkstra(G=self.G, source=o_uid)
        except Exception as e:
            print(f"\033[31m[ERROR] Dijkstra failed for origin {o_uid}: {e}\033[0m")
            return path_list, pathvia_list

        for d_uid in uid_list:
            if o_uid == d_uid or d_uid not in s_paths:
                continue

            shortest_path = s_paths[d_uid]
            shortest_path_length = s_lengths[d_uid]
            max_path_length = min(shortest_path_length * (1 + theta1), shortest_path_length + theta2)

            lengths, paths = self.find_k_paths_via_yen(
                shortest_path=shortest_path,
                shortest_path_length=shortest_path_length,
                max_length=max_path_length
            )

            transfers = [self.get_trans_cnt(pa) for pa in paths]
            if not transfers:
                continue

            max_transfers = min(transfers) + transfer_deviation
            k_ = 0
            base_path_id = int(f"{o_uid}{d_uid}00")
            for transfer, length, path in zip(transfers, lengths, paths):
                if transfer <= max_transfers:
                    k_ += 1
                    path_id = base_path_id + k_
                    path_list.append([
                        path_id, length, transfer,
                        "_".join(self.compress_passing_info(path=path))
                    ])
                    pv = self.get_path_via(path=path, path_id=path_id)
                    pathvia_list.extend(pv)
                if k_ >= k:
                    break

        return path_list, pathvia_list
    
    
if __name__ == "__main__":
    config.load_config()
    net = ChengduMetro(nodes=gen_node_from_sta(), links=gen_links())
    net.plot_metro_net(coordinates=read_("coordinates.csv", show_timer=False))
    pass
