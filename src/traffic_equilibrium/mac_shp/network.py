import logging
import os
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Iterable, Optional, Mapping
import time

import fiona
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import Point, LineString, Polygon, shape
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from toolz import valmap, curry
from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips

from .demand import read_csv, to_demand, Demand
from .features import to_node, to_links
from .link import Link
from .node import Coordinate, Node
from .schemas import LinkAttributes, Feature
from .spatial_index import SpatialIndex, insert_point, nearest_neighbor
from enum import Enum

log = logging.getLogger(__name__)
DEFAULT_NODE_FILE = 'node.shp'
DEFAULT_LINK_FILE = 'link.shp'
ZONES = 'zones'
LINK_KEY = 'link'


class MalformedNetworkException(Exception):
    pass


class NoNodeFoundException(MalformedNetworkException):
    pass


class ThruTrips(Enum):
    NoThruTrips = 1       # all trips MUST be on the subgraph
    PartialThruTrips = 2  # all trips MUST have >=1 endpoint on the subgraph
    AllThruTrips = 3      # trips must intersect the subgraph

    def is_valid_trip(self, orgn, orgn_pt, dest, dest_pt,
                      geofence, subgraph) -> bool:
        orgn_on_subgraph = orgn in subgraph
        dest_on_subgraph = dest in subgraph
        if self is ThruTrips.NoThruTrips:
            # valid iff both end points are on subgraph
            return orgn_on_subgraph and dest_on_subgraph
        is_partial = orgn_on_subgraph or dest_on_subgraph
        if self is ThruTrips.PartialThruTrips:
            # valid iff >=1 endpoint is on subgraph
            return is_partial
        if self is ThruTrips.AllThruTrips:
            # valid iff >=1 endpoint is on subgraph
            # OR trip (as a straight line) intersects geofence
            return is_partial or is_thru_trip(geofence, orgn_pt, dest_pt)


def shp_items(shp_file: str):
    return fiona.open(shp_file).values()


def nearest_coordinate(idx: SpatialIndex, pt: Coordinate) -> Node:
    x, y = pt
    result = nearest_neighbor(idx, x, y)
    if result is None:
        raise NoNodeFoundException
    else:
        return result.object


def geofence_contains_link(geofence: Polygon, link: Link) -> bool:
    return geofence.intersects(shape(link.geometry))


def is_thru_trip(geofence: Polygon, origin: Point, destination: Point) -> bool:
    return LineString([origin, destination]).intersects(geofence)


class NetworkBuilder:
    def __init__(self):
        self.node_index = SpatialIndex()
        self.graph = nx.OrderedDiGraph()
        self.trips = nx.OrderedDiGraph()
        self.zone_links = defaultdict(list)
        self._zone_nodes = None

    def read_shp(self, directory: str,
                 node_file: str = None, link_file: str = None):

        self.read_node_shp(os.path.join(directory,
                                        node_file or DEFAULT_NODE_FILE))
        self.read_link_shp(os.path.join(directory,
                                        link_file or DEFAULT_LINK_FILE))
        return self

    def read_od_matrix(self, directory: str):
        od_matrix = sum(map(read_csv, _csv_files(directory)))
        log.info(
            f"Read {od_matrix.shape} OD matrix with {np.count_nonzero(od_matrix)} nonzero od pair volumes.")
        self.add_trips_from_od_matrix(od_matrix)
        return self

    def read_node_shp(self, node_file: str):
        nodes_read = 0
        for i, node_data in enumerate(shp_items(node_file)):
            node = to_node(i, node_data[Feature.properties])
            self.add_node(node)
            nodes_read += 1
        log.info(f"Read {nodes_read} nodes from {node_file}.")

    def read_link_shp(self, link_file: str):
        links_read = 0
        for edge_pair in shp_items(link_file):
            links = to_links(edge_pair[Feature.properties],
                             edge_pair[Feature.geometry])
            for link in links:
                if link.is_link:
                    self.add_link(link)
                    links_read += 1
        log.info(f"Read {links_read} links from {link_file}.")

    def add_node(self, n: Node):
        insert_point(self.node_index,
                     n.id, *n.coordinate, obj=n)
        self.graph.add_node(n.id, data=n)

    def add_link(self, link: Link):
        _from_node = nearest_coordinate(self.node_index,
                                        link.from_point)
        _to_node = nearest_coordinate(self.node_index,
                                      link.to_point)
        e = (_from_node.id, _to_node.id)
        self.graph.add_edge(*e, data=link)
        if link.is_virtual:
            self.zone_links[link.zone].append(e)

    @property
    def zone_nodes(self):
        if self._zone_nodes is None:
            self._zone_nodes = valmap(extract_virtual_node(self.graph),
                                      self.zone_links)
        return self._zone_nodes

    @staticmethod
    def _is_valid_trip(origin: int, destination: int, volume: float) -> bool:
        return (
                origin is not None
                and destination is not None
                and origin != destination
                and volume > 0.0
        )

    def add_trip(self, trip: Demand):
        origin = self.zone_nodes.get(trip.from_zone)
        destination = self.zone_nodes.get(trip.to_zone)
        if self._is_valid_trip(origin, destination, trip.volume):
            if (origin, destination) not in self.trips.edges:
                self.trips.add_edge(origin, destination, volume=0.0)
            self.trips.edges[origin, destination]['volume'] += trip.volume

    def add_trips_from_od_matrix(self, od_matrix: np.array):
        for trip in to_demand(od_matrix):
            self.add_trip(trip)

    def nodes_within(self, geofence: Polygon) -> Iterable[Node]:
        for node in self.node_index.intersection(geofence.bounds,
                                                 objects='raw'):
            if geofence.contains(node.point):
                yield node

    def connected_subgraph_by_geofence(self, geofence: Polygon) \
            -> nx.OrderedDiGraph:
        t0 = time.time()
        node_ids = sorted(n.id for n in self.nodes_within(geofence))
        subgraph = largest_strongly_connected_subgraph(
            self.graph.subgraph(node_ids)
        )
        subgraph.graph['geofence'] = geofence
        log.info(f"Extracted strongly connected geofenced subgraph in {time.time() - t0:0.2f} seconds.")
        return subgraph

    def snap_trips_to_subgraph(self, subgraph: nx.OrderedDiGraph,
                               thru_trips: ThruTrips = ThruTrips.AllThruTrips) \
            -> nx.OrderedDiGraph:
        log.info("Snapping trips to subgraph.")
        node_idx = SpatialIndex(
            (n_id, d['data'].point.bounds, d['data'])
            for n_id, d in subgraph.nodes.data()
        )
        geofence = subgraph.graph['geofence']
        trips = nx.OrderedDiGraph()
        t0 = time.time()
        for orgn, dest, data in self.trips.edges.data():
            orgn_pt = self.graph.nodes[orgn]['data'].coordinate
            dest_pt = self.graph.nodes[dest]['data'].coordinate
            if thru_trips.is_valid_trip(orgn, orgn_pt, dest, dest_pt,
                                        geofence, subgraph):
                if orgn not in subgraph:
                    orgn = nearest_coordinate(node_idx, orgn_pt).id
                if dest not in subgraph:
                    dest = nearest_coordinate(node_idx, dest_pt).id
                if (orgn, dest) not in trips.edges:
                    trips.add_edge(orgn, dest, **data)
                else:
                    trips.edges[orgn, dest]['volume'] += data['volume']
        log.info(f"Snapped trips to subgraph in {time.time() - t0:0.2f} seconds.")
        return trips

    def extract_by_geofence(self, geofence: Polygon,
                            thru_trips: ThruTrips = ThruTrips.AllThruTrips) \
            -> Tuple[nx.OrderedDiGraph, nx.OrderedDiGraph]:
        subgraph = self.connected_subgraph_by_geofence(geofence)
        trips = self.snap_trips_to_subgraph(subgraph, thru_trips)
        return reindex(subgraph, trips)

    def to_network(self, name: str, demand_threshold: float = 0.0,
                   demand_multiplier: float = 1.0,
                   geofence: Optional[Polygon] = None,
                   thru_trips: ThruTrips = ThruTrips.AllThruTrips) -> Tuple[DiGraph, Trips]:
        if geofence is not None:
            log.info(f"Extracting a subnetwork by geofence.")
            graph, trips = self.extract_by_geofence(geofence, thru_trips)
        else:
            graph = self.graph
            trips = self.trips
        log.info(
            f"Creating network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} links.")
        network = DiGraph(name)
        network.append_nodes(graph.number_of_nodes())
        edata = list(graph.edges.data())
        network.add_links_from([(u, v) for u, v, _ in edata])
        network.set_link_info({i: d['data']
                               for i, (_, _, d) in enumerate(edata)})
        log.info("Creating demand...")
        demand = Trips()
        for s, t, data in trips.edges.data():
            volume = data['volume']
            if volume > demand_threshold:
                demand.append(s, t, demand_multiplier * volume)
        log.info(
            f"Created demand with {demand.number_of_trips()} trips and {demand.total_volume()} total volume.")
        return network, demand


def largest_strongly_connected_subgraph(graph: nx.OrderedDiGraph) \
        -> nx.OrderedDiGraph:
    node_ids = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(node_ids)


def reindex(graph: nx.OrderedDiGraph, trips: nx.OrderedDiGraph) \
        -> Tuple[nx.OrderedDiGraph, nx.OrderedDiGraph]:
    lookup = {
        n: i for (i, n) in enumerate(graph.nodes)
    }
    return _reindex(lookup, graph), _reindex(lookup, trips)


def _reindex(idx_lookup: dict, graph: nx.OrderedDiGraph) -> nx.OrderedDiGraph:
    new_graph = nx.OrderedDiGraph(**graph.graph)
    for u, data in graph.nodes.data():
        new_graph.add_node(idx_lookup[u], **data)
    for u, v, data in graph.edges.data():
        new_graph.add_edge(idx_lookup[u], idx_lookup[v], **data)
    return new_graph


def distance_miles(a, b) -> float:
    return geodesic(a, b).miles


def geodesic_affinity(X):
    return pairwise_distances(X, metric=distance_miles)


def cluster_zones(zone_nodes: Mapping[int, int],
                  node_pts: Mapping[int, Coordinate],
                  distance_threshold: float):
    zone_pts = {zone_id: node_pts[node_id]
                for zone_id, node_id in zone_nodes.items()
                if node_id is not None}
    zones, pts = zip(*zone_pts.items())
    data = np.array(pts)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        affinity=geodesic_affinity,
        linkage="average",
    ).fit(data)
    n_clusters = clustering.n_clusters_
    zone_labels = clustering.labels_
    print(f"Found {n_clusters} aggregated_zones.")
    zone_node_centroids = np.zeros(len(zones), dtype=np.int)
    for i in range(n_clusters):
        cluster_mask = zone_labels == i
        idx, = cluster_mask.nonzero()
        cluster_pts = data[cluster_mask]
        centroid = cluster_pts.mean(0)
        center_node = idx[abs(cluster_pts - centroid).sum(1).argmin()]
        zone_node_centroids[cluster_mask] = center_node
    return OrderedDict(zip(zones, zone_node_centroids))


def _csv_files(directory: str) -> Iterable[str]:
    for fname in os.listdir(directory):
        if fname.endswith('.csv'):
            yield os.path.join(directory, fname)


@curry
def extract_virtual_node(network: nx.DiGraph,
                         virtual_links: List[Tuple[int, int]]) -> Optional[int]:
    it = iter(virtual_links)
    s = set(next(it))
    for l in it:
        s.intersection_update(l)
    n_candidates = len(s)
    if n_candidates == 1:
        return s.pop()
    elif n_candidates == 0:
        log.warning("Could not resolve a virtual node, returning None")
        return None
    else:
        _, n = min(
            (network.out_degree(n), n) for n in s
        )
        return n


def array_of(data, key: str) -> np.ndarray:
    a = np.zeros(len(data))
    for i, row in data.items():
        if isinstance(row, dict):
            row = Link(**row)
        a[int(i)] = getattr(row, key)
    return a


def get_attr_or_field(item, key):
    try:
        return getattr(item, key)
    except AttributeError:
        pass
    try:
        return item[key]
    except KeyError:
        print(item)
        raise

def to_free_flow_travel_time(network: DiGraph):
    return array_of(network.link_info, LinkAttributes.free_flow)


def to_capacity(network: DiGraph):
    return array_of(network.link_info, LinkAttributes.capacity)
