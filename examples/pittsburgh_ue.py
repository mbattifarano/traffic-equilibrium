import os
import logging

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem
from traffic_equilibrium.solver import FrankWolfeSettings, solve

logging.basicConfig(level=logging.INFO)

problem = to_equilibrium_problem(os.path.join(
    'test', 'fixtures', 'pittsburgh-small-network'),
    'nodes.shp',
    'links.shp'
)
print(problem.network.info())

settings = FrankWolfeSettings(
    50,
    1e-6,
    1e-10
)

if __name__ == '__main__':
    print("Solving pittsburgh ue")
    result = solve(problem, settings)
    print(f"Solved pittsburgh ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
