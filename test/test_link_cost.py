from traffic_equilibrium.link_cost import LinkCostBPR
from traffic_equilibrium.vector import Vector
import numpy as np


def test_init_from_vectors():
    n = 10
    cap = Vector.copy_of(np.ones(n).astype(np.double))
    fftt = Vector.copy_of(np.ones(n).astype(np.double))
    flow = Vector.copy_of(np.zeros(n).astype(np.double))
    bpr = LinkCostBPR(1.0, 1.0, cap, fftt)
    cost = bpr.compute_link_cost_vector(flow)
    assert len(cost) == n
    assert np.allclose(list(cost), 1.0)
