from pytest import fixture
from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips
from traffic_equilibrium.link_cost import LinkCostLinear
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.solver import Problem
import numpy as np

@fixture
def braess_network():
    """Braess Network
            1
          ↗ ⏐ ↘
        0   ⏐   3
          ↘ ↓ ↗
            2
    """
    net = DiGraph("braess")
    net.append_nodes(4)
    net.add_links_from([
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
    ])
    return net


@fixture
def braess_trips():
    trips = Trips()
    trips.append(0, 3, 6.0)
    return trips


@fixture
def braess_cost_function():
    coefficients = Vector.copy_of(np.array([10, 1, 1, 1, 10], np.double))
    constants = Vector.copy_of(np.array([0, 50, 10, 50, 0], np.double))
    return LinkCostLinear(coefficients, constants)


@fixture
def braess_problem(braess_network, braess_trips, braess_cost_function):
    return Problem(
        braess_network,
        braess_trips.compile(),
        braess_cost_function
    )
