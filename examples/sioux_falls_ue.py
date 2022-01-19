import os
import time
import warnings
import numpy as np
from traffic_equilibrium.tntp import (
    TNTPNetwork, TNTPTrips, TNTPSolution
)
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import Problem, FrankWolfeSettings, solve

TNTP_DIRECTORY = os.path.join(
    'test',
    'fixtures',
    'TransportationNetworks',
    'SiouxFalls'
)
demand_scales = [
    0.01,
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0,
]
demand_scales = [1.0]

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
    500 * 1000,
    1e-5,
    1e-10,
    10000
)

paths = PathDB(os.path.join("examples", "results", "sioux-falls-paths.db"))

if __name__ == '__main__':
    for demand_scale in demand_scales:
        tntp_scaled_trips = tntp_trips.scale_volume(demand_scale)
        problem = Problem(
            tntp_net.to_road_network(),
            tntp_scaled_trips.to_trips(tntp_net).compile(),
            tntp_net.to_link_cost_function()
        )
        print(f"Solving sioux falls UE. (demand scale: {demand_scale})")
        result = solve(problem, settings, paths)
        print(
            f"Solved pittsburgh so to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
        if demand_scale == 1.0:
            actual_flow = result.flow.to_array()
            error = abs(actual_flow - expected_flow)
            print(f"link flow error vs expected result: {error.mean()} avg, {error.max()} max")
        print("Saving results")
        t0 = time.time()
        result.save(f'examples/results/sioux-falls-ue-scale-{demand_scale}')
        print(f"saved results in {time.time()-t0} seconds.")

