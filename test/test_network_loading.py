import pytest
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.network_loading import load_network
import numpy as np


def test_braess_cost_fn(braess_cost_function):
    zero = Vector.copy_of(np.zeros(5))
    free_flow_cost = braess_cost_function.compute_link_cost_vector(zero)
    assert np.allclose(free_flow_cost.to_array(),
                       np.array([0, 50, 10, 50, 0]))
    eq_flow = Vector.copy_of(np.array([4, 2, 2, 2, 4], np.double))
    eq_cost = braess_cost_function.compute_link_cost_vector(eq_flow)
    assert np.allclose(eq_cost.to_array(),
                       np.array([40, 52, 12, 52, 40]))
    assert np.dot(eq_flow.to_array(), eq_cost.to_array()) == pytest.approx(92*6)
    so_flow = Vector.copy_of(np.array([3, 3, 0, 3, 3], np.double))
    so_cost = braess_cost_function.compute_link_cost_vector(so_flow)
    assert np.allclose(so_cost.to_array(),
                       np.array([30, 53, 10, 53, 30]))
    assert np.dot(so_flow.to_array(), so_cost.to_array()) == pytest.approx(83*6)


def test_braess_network_loading(braess_network, braess_trips, braess_cost_function):
    zero = Vector.copy_of(np.zeros(5))
    free_flow_cost = braess_cost_function.compute_link_cost_vector(zero)
    demand = braess_trips.compile()
    flow, paths = load_network(braess_network, free_flow_cost, demand)
    _flow = flow.to_array()
    assert np.allclose(_flow, np.array([6, 0, 6, 0, 6]))



