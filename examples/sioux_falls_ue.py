import os
import warnings
import numpy as np
from traffic_equilibrium.tntp import (
    TNTPNetwork, TNTPTrips, TNTPSolution
)
from traffic_equilibrium.solver import Problem, FrankWolfeSettings, solve

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

settings = FrankWolfeSettings(
    1000000,
    1e-5,
    1e-8,
)

print("Solving Sioux Falls UE")
result = solve(problem, settings)
print(f"Solved Sioux Falls ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
flow = result.flow.to_array()
cost = result.cost.to_array()
assert np.linalg.norm((flow - expected_flow)/expected_flow) <= 0.01
assert flow.dot(expected_cost) / expected_flow.dot(expected_cost) - 1.0 <= 1e-3
