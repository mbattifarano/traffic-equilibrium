import os
import warnings
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Iterable, Optional
from dataclasses import dataclass

import fiona
import numpy as np
from toolz import valmap, curry

from .spatial_index import SpatialIndex, insert_point, nearest_neighbor

from .node import Coordinate, Node
from .schemas import LinkAttributes, Feature
from .features import to_node, to_links
from .demand import read_csv, to_demand

from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips

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
                          node_file: str = DEFAULT_NODE_FILE,
                          link_file: str = DEFAULT_LINK_FILE,
                          ) -> NetworkData:
    net_name = os.path.basename(directory)
    network = DiGraph(net_name)
    idx = SpatialIndex()
    zones = defaultdict(list)
    links = OrderedDict()
    number_of_nodes = 0
    for i, node in enumerate(shp_items(os.path.join(directory, node_file))):
        # nodes are indexed sequentially
        n = to_node(i, node[Feature.properties])
        insert_point(idx, n.id, *n.coordinate, obj=n)
        number_of_nodes += 1
    network.append_nodes(number_of_nodes)
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
    network.add_links_from(list(links.keys()))
    zone_nodes = valmap(extract_virtual_node(network), zones)
    return NetworkData(
        network,
        links,
        zone_nodes
    )


def _csv_files(directory: str) -> Iterable[str]:
    for fname in os.listdir(directory):
        if fname.endswith('.csv'):
            yield os.path.join(directory, fname)


def _get_virtual_node(network: NetworkData,
                      zone_id: int) -> Optional[int]:
    return network.zone.get(zone_id)


def travel_demand(network: NetworkData, directory: str) -> Trips:
    od_matrix = sum(map(read_csv, _csv_files(directory)))
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
    return trips


@curry
def extract_virtual_node(network: DiGraph,
                         virtual_links: List[Tuple[int, int]]) -> Optional[int]:
    ns = set()
    for l in virtual_links:
        ns.update(l)
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


def edges(graph):
    return sorted(graph.edges)


def array_of(data, key: str) -> np.ndarray:
    return np.array([
        getattr(row, key)
        for row in data.values()
    ])


def to_free_flow_travel_time(network: NetworkData):
    return array_of(network.links, LinkAttributes.free_flow)


def to_capacity(network: NetworkData):
    return array_of(network.links, LinkAttributes.capacity)
