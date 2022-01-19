import os
import warnings
import numpy as np
from enum import Enum

from shapely.geometry import Polygon
from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.link_cost import LinkCostBPR, LinkCostMarginalBPR
from traffic_equilibrium.vector import Vector
from .network import (
    NetworkBuilder,
    to_free_flow_travel_time,
    to_capacity,
    ThruTrips,
)

DEMAND_DIR = "ODmatrix"


class ProblemMode(Enum):
    USER_EQUILIBRIUM = 1
    SYSTEM_OPTIMAL = 2



def _bpr(network: DiGraph, mode: ProblemMode):
    cap = to_capacity(network).astype(np.double)
    if cap.min() <= 0:
        warnings.warn("Capacity has minimum value %g." % cap.min())
    freeflow = to_free_flow_travel_time(network).astype(np.double)
    cls = {
        ProblemMode.USER_EQUILIBRIUM: LinkCostBPR,
        ProblemMode.SYSTEM_OPTIMAL: LinkCostMarginalBPR,
    }[mode]
    return cls(0.15, 4.0, Vector.copy_of(cap), Vector.copy_of(freeflow))


def to_equilibrium_problem(directory: str,
                           node_file: str = None,
                           link_file: str = None,
                           mode: ProblemMode = ProblemMode.USER_EQUILIBRIUM,
                           demand_threshold: float = 0.0,
                           demand_multiplier: float = 1.0,
                           geofence: Polygon = None,
                           thru_trips: ThruTrips = ThruTrips.AllThruTrips,
                           ) -> Problem:
    name = os.path.basename(directory)
    network, trips = (NetworkBuilder()
                      .read_shp(directory, node_file, link_file)
                      .read_od_matrix(os.path.join(directory, DEMAND_DIR))
                      .to_network(name, demand_threshold, demand_multiplier,
                                  geofence, thru_trips)
                      )
    cost_fn = _bpr(network, mode)
    return Problem(
        network,
        trips.compile(),
        cost_fn
    )
