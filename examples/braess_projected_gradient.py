import numpy as np
import os

from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.trips import Trips
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.link_cost import LinkCostLinear
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.projected_gradient import ProjectedGradientSettings, ProjectedGradientResult, PathSet

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


settings = ProjectedGradientSettings(
    1e-2,
    1e-6,
    3,
    10,
    1e-4,
    1
)

path_set = PathSet.create_from_problem(problem)

result = ProjectedGradientResult(problem, path_set, settings)
print("Solving braess ue")
result.solve_via_projected_gradient()
