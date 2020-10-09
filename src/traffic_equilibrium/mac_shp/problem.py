import os
import warnings
import numpy as np
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.link_cost import LinkCostBPR
from traffic_equilibrium.vector import Vector
from .network import (
    NetworkData,
    network_data_from_shp,
    travel_demand,
    to_free_flow_travel_time,
    to_capacity
)

DEMAND_DIR = "ODmatrix"


def _bpr(network_data: NetworkData):
    cap = to_capacity(network_data).astype(np.double)
    if cap.min() <= 0:
        warnings.warn("Capacity has minimum value %g." % cap.min())
    freeflow = to_free_flow_travel_time(network_data).astype(np.double)
    return LinkCostBPR(0.15, 4.0, Vector.copy_of(cap), Vector.copy_of(freeflow))


def to_equilibrium_problem(directory: str) -> Problem:
    network_data = network_data_from_shp(directory)
    cost_fn = _bpr(network_data)
    trips = travel_demand(network_data, os.path.join(directory, DEMAND_DIR))
    return Problem(
        network_data.network,
        trips.compile(),
        cost_fn
    )
