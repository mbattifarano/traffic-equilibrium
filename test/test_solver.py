import warnings
from traffic_equilibrium.solver import FrankWolfeSettings, solve

import numpy as np


def test_braess_ue(braess_problem):
    settings = FrankWolfeSettings(
        100,
        1e-12,
        1e-8,
    )
    assert settings.max_iterations == 100
    assert settings.gap_tolerance == 1e-12
    assert settings.line_search_tolerance == 1e-8

    result = solve(braess_problem, settings)
    assert result.iterations >= 0
    assert result.iterations <= 100
    assert (result.iterations == 100) or (result.gap < 1e-12)
    flow = result.flow.to_array()
    cost = result.cost.to_array()
    assert np.allclose(flow, [4, 2, 2, 2, 4])
    assert np.allclose(cost, [40, 52, 12, 52, 40])
    warnings.warn(f"Solved braess ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds.")

