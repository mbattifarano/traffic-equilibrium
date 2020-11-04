import os
from pytest import fixture
from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips
from traffic_equilibrium.link_cost import LinkCostLinear
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.tntp.network import TNTPNetwork
from traffic_equilibrium.tntp.trips import TNTPTrips
from traffic_equilibrium.tntp.solution import TNTPSolution
from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem
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


@fixture
def sioux_falls_problem():
    with open('test/fixtures/TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp') as fp:
        tntp_net = TNTPNetwork.read_file('SiouxFalls', fp)
    with open('test/fixtures/TransportationNetworks/SiouxFalls/SiouxFalls_trips.tntp') as fp:
        tntp_trips = TNTPTrips.read_file(fp)
    return Problem(
        tntp_net.to_road_network(),
        tntp_trips.to_trips(tntp_net).compile(),
        tntp_net.to_link_cost_function(),
    )


@fixture
def sioux_falls_solution():
    with open('test/fixtures/TransportationNetworks/SiouxFalls/SiouxFalls_flow.tntp') as fp:
        tntp_solution = TNTPSolution.read_text(fp.read())
    return tntp_solution


@fixture(scope='session')
def pittsburgh_problem():
    return to_equilibrium_problem(os.path.join('test', 'fixtures',
                                               'pittsburgh-small-network'),
                                  'nodes.shp',
                                  'links.shp'
                                  )
