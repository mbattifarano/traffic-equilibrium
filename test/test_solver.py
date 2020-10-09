import warnings
from traffic_equilibrium.solver import FrankWolfeSettings, solve

import numpy as np


def test_braess_ue(braess_problem):
    settings = FrankWolfeSettings(
        1000,
        1e-6,
        1e-6,
    )
    result = solve(braess_problem, settings)
    warnings.warn(f"Solved braess ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds.")
    assert result.iterations >= 0
    assert result.iterations <= settings.max_iterations
    assert (result.iterations == settings.max_iterations) or (result.gap < settings.gap_tolerance)
    flow = result.flow.to_array()
    cost = result.cost.to_array()
    assert np.allclose(flow, [4, 2, 2, 2, 4])
    assert np.allclose(cost, [40, 52, 12, 52, 40])


def test_sioux_falls_ue(sioux_falls_problem, sioux_falls_solution):
    expected_flow = sioux_falls_solution.link_flow()
    expected_cost = sioux_falls_solution.link_cost()

    settings = FrankWolfeSettings(
        1000000,
        1e-5,
        1e-8,
    )
    print("Solving SiouxFalls UE")
    result = solve(sioux_falls_problem, settings)
    warnings.warn(f"Solved Sioux Falls ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
    flow = result.flow.to_array()
    cost = result.cost.to_array()
    assert np.linalg.norm((flow - expected_flow)/expected_flow) <= 0.01
    assert flow.dot(expected_cost) / expected_flow.dot(expected_cost) - 1.0 <= 1e-3
