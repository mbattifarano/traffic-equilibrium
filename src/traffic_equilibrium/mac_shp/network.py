import os
import warnings
import logging
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Iterable, Optional
from dataclasses import dataclass

import fiona
import numpy as np
import networkx as nx
from toolz import valmap, curry

from .spatial_index import SpatialIndex, insert_point, nearest_neighbor

from .node import Coordinate, Node
from .schemas import LinkAttributes, Feature
from .features import to_node, to_links
from .demand import read_csv, to_demand

from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips


log = logging.getLogger(__name__)
DEFAULT_NODE_FILE = 'node.shp'
DEFAULT_LINK_FILE = 'link.shp'
ZONES = 'zones'
LINK_KEY = 'link'


@dataclass
class NetworkData:
    network: DiGraph
    links: dict
    zones: dict


class MalformedNetworkException(Exception):
    pass


class NoNodeFoundException(MalformedNetworkException):
    pass


def shp_items(shp_file: str):
    return fiona.open(shp_file).values()


def nearest_coordinate(idx: SpatialIndex, pt: Coordinate) -> Node:
    x, y = pt
    result = nearest_neighbor(idx, x, y)
    if result is None:
        raise NoNodeFoundException
    else:
        return result.object


def network_data_from_shp(directory: str,
                          node_file: str = None,
                          link_file: str = None,
                          ) -> NetworkData:
    log.info(f"Reading shp files from {directory}.")
    node_file = node_file or DEFAULT_NODE_FILE
    link_file = link_file or DEFAULT_LINK_FILE
    net_name = os.path.basename(directory)
    network = DiGraph(net_name)
    g = nx.DiGraph()
    idx = SpatialIndex()
    zones = defaultdict(list)
    links = OrderedDict()
    for i, node in enumerate(shp_items(os.path.join(directory, node_file))):
        # nodes are indexed sequentially
        n = to_node(i, node[Feature.properties])
        insert_point(idx, n.id, *n.coordinate, obj=n)
        g.add_node(n.id, data=n)
    log.info(f"Read {len(g)} nodes from {node_file}.")
    for edge_pair in shp_items(os.path.join(directory, link_file)):
        for link in to_links(edge_pair[Feature.properties],
                             edge_pair[Feature.geometry]):
            if link.is_link:
                _from_node = nearest_coordinate(idx, link.from_point)
                _to_node = nearest_coordinate(idx, link.to_point)
                e = (_from_node.id, _to_node.id)
                if link.is_virtual:
                    zones[link.zone].append(e)
                # links is ordered
                # links will be added to network in this order
                links[e] = link
                g.add_edge(*e, data=link)
    log.info(f"Read {g.number_of_edges()} links from {link_file}.")
    log.info(f"Computing connected components...")
    largest_cc_node_ids = max(nx.strongly_connected_components(g), key=len)
    number_of_nodes = len(largest_cc_node_ids)
    log.info(f"Largest connected components has {number_of_nodes} nodes.")
    node_id_map = {n: i for (i, n) in enumerate(largest_cc_node_ids)}
    network.append_nodes(number_of_nodes)
    cc_links = OrderedDict()
    for (u, v), link in links.items():
        if (u in node_id_map) and (v in node_id_map):
            u_i = node_id_map[u]
            v_i = node_id_map[v]
            cc_links[(u_i, v_i)] = link
    cc_zones = {}
    for zone_id, old_links in zones.items():
        new_links = []
        for old_u, old_v in old_links:
            if old_u in node_id_map and old_v in node_id_map:
                new_links.append((node_id_map[old_u], node_id_map[old_v]))
        if new_links:
            cc_zones[zone_id] = new_links
    network.add_links_from(list(cc_links.keys()))
    zone_nodes = valmap(extract_virtual_node(network), cc_zones)
    print(f"number of zone nodes: {len(zone_nodes)}")
    log.info(f"Created igraph road network: {network.info()}")
    return NetworkData(
        network,
        cc_links,
        zone_nodes
    )


def _csv_files(directory: str) -> Iterable[str]:
    for fname in os.listdir(directory):
        if fname.endswith('.csv'):
            yield os.path.join(directory, fname)


def travel_demand(network: NetworkData, directory: str) -> Trips:
    od_matrix = sum(map(read_csv, _csv_files(directory)))
    print(f"Recovered {od_matrix.shape} OD matrix with {np.count_nonzero(od_matrix)} nonzero od pair volumes.")
    trips = Trips()
    for d in to_demand(od_matrix):
        origin = network.zones.get(d.from_zone)
        destination = network.zones.get(d.to_zone)
        if (origin is not None
                and destination is not None
                and d.volume > 0
                and origin != destination):
            trips.append(
                origin,
                destination,
                d.volume
            )
    print(f"Found {len(trips.trips)} OD pairs.")
    return trips


@curry
def extract_virtual_node(network: DiGraph,
                         virtual_links: List[Tuple[int, int]]) -> Optional[int]:
    it = iter(virtual_links)
    s = set(next(it))
    for l in it:
        s.intersection_update(l)
    n_candidates = len(s)
    if n_candidates == 1:
        return s.pop()
    elif n_candidates == 0:
        warnings.warn("Could not resolve a virtual node, returning None")
        return None
    else:
        _, n = min(
            (network.degree_of(n), n) for n in s
        )
        return n


def array_of(data, key: str) -> np.ndarray:
    return np.array([
        getattr(row, key)
        for row in data.values()
    ])


def to_free_flow_travel_time(network: NetworkData):
    return array_of(network.links, LinkAttributes.free_flow)


def to_capacity(network: NetworkData):
    return array_of(network.links, LinkAttributes.capacity)
