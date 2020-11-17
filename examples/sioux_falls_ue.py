import os
import time
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

print(problem.network.info())

settings = FrankWolfeSettings(
    100000,
    1e-6,
    1e-10,
)

if __name__ == '__main__':
    print("Solving sioux falls ue")
    result = solve(problem, settings)
    print(f"Solved pittsburgh ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
    print("Saving results")
    t0 = time.time()
    result.save('examples/results/sioux-falls-ue', 0)
    print(f"saved results in {time.time()-t0} seconds.")
