import numpy as np
import os

from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.link_cost import LinkCostLinear
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import Problem, FrankWolfeSettings, solve

net = DiGraph("braess")
net.append_nodes(4)
net.add_links_from([
    (0, 1),  # 0
    (0, 2),  # 1
    (1, 2),  # 2
    (1, 3),  # 3
    (2, 3),  # 4
])

trips = Trips()
trips.append(0, 3, 6.0)

coefficients = Vector.copy_of(np.array([10, 1, 1, 1, 10], np.double))
constants = Vector.copy_of(np.array([0, 50, 10, 50, 0], np.double))
cost_fn = LinkCostLinear(coefficients, constants)

problem = Problem(
    net,
    trips.compile(),
    cost_fn
)

settings = FrankWolfeSettings(
    1000,
    1e-6,
    1e-8,
    100
)

paths = PathDB(os.path.join("examples/results/braess-paths.db"))

print("Solving braess ue")
result = solve(problem, settings, paths)
print(f"Solved braess ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
print(f"link flow = {result.flow.to_array()}")
print(f"link cost = {result.cost.to_array()}")
print("saving results")
result.save(os.path.join("examples", "results", "braess-ue"))
