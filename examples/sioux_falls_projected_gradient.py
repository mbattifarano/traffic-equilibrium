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
    tntp_net.to_link_cost_function(),
)

print(problem.network.info())

settings = ProjectedGradientSettings(
    1e-10,  # rgap tol
    0.0,  # mec tol
    0.0,   # aec tol
    1000,   # max iterations
    1e-16,   # line search tolerance
    10,    # report interval
)

paths = PathDB(os.path.join("examples", "results", "sioux-falls-paths-pg.db"))

path_set = PathSet.create_from_problem(problem, paths)

result = ProjectedGradientResult(problem, path_set, settings)


if __name__ == '__main__':
    result.solve_via_projected_gradient()
    actual_flow = result.path_set.to_link_flow_numpy(len(expected_flow))
    print(f"max abs link flow error: {np.max(np.abs(expected_flow - actual_flow))} ({np.max(np.abs(expected_flow - actual_flow)/actual_flow)}%)")
    assert np.allclose(expected_flow, actual_flow)
    result.save('examples/results/sioux-falls-path-based-ue')
