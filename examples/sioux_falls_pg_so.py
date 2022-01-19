import os
from traffic_equilibrium.tntp import (
    TNTPNetwork, TNTPTrips, TNTPSolution
)
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.projected_gradient import ProjectedGradientResult, ProjectedGradientSettings, PathSet
from traffic_equilibrium.pathdb import PathDB

import numpy as np

TNTP_DIRECTORY = os.path.join(
    'test',
    'fixtures',
    'TransportationNetworks',
    'SiouxFalls'
)

with open(os.path.join(TNTP_DIRECTORY, 'SiouxFalls_net.tntp')) as fp:
    tntp_net = TNTPNetwork.read_text('SiouxFalls', fp.read())

with open(os.path.join(TNTP_DIRECTORY, 'SiouxFalls_trips.tntp')) as fp:
    tntp_trips = TNTPTrips.read_text(fp.read())

with open(os.path.join(TNTP_DIRECTORY, 'SiouxFalls_flow.tntp')) as fp:
    tntp_solution = TNTPSolution.read_text(fp.read())

expected_flow = tntp_solution.link_flow()
expected_cost = tntp_solution.link_cost()

problem = Problem(
    tntp_net.to_road_network(),
    tntp_trips.to_trips(tntp_net).compile(),
    tntp_net.to_marginal_link_cost_function(),
)

print(problem.network.info())

settings = ProjectedGradientSettings(
    1e-7,  # rgap tol
    1e-6,   # mec tol
    1e-7,  # aec tol
    10000,   # max iterations
    1e-12,   # line search tolerance
    100,    # report interval
)

paths = PathDB(os.path.join("examples", "results", "sioux-falls-paths-pg.db"))

path_set = PathSet.create_from_problem(problem, paths)

result = ProjectedGradientResult(problem, path_set, settings)


if __name__ == '__main__':
    result.solve_via_projected_gradient()
    result.save('examples/results/sioux-falls-path-based-so')
