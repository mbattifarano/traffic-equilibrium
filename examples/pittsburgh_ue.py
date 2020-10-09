import os

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem
from traffic_equilibrium.solver import FrankWolfeSettings, solve

problem = to_equilibrium_problem(os.path.join(
    'test', 'fixtures', 'pittsburgh-network'
))

settings = FrankWolfeSettings(
    5,
    1e-12,
    1e-8
)

print("Solving pittsburgh ue")
result = solve(problem, settings)
print(f"Solved pittsburgh ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
