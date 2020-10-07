from traffic_equilibrium.vector import Vector
import numpy as np


def test_numpy_view():
    n = 4
    a = np.arange(n).astype(np.double)
    vector = Vector.view_of(a)
    assert len(vector) == n
    assert list(vector) == list(a) == list(range(n))
