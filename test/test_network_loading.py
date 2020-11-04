import pytest
import time
import warnings
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
    flow, paths, best_path_cost = load_network(braess_network, free_flow_cost, demand)
    _flow = flow.to_array()
    assert np.allclose(best_path_cost.to_array(), np.array([10.0]))
    assert np.allclose(_flow, np.array([6, 0, 6, 0, 6]))


def test_sioux_falls_network_loading(sioux_falls_problem):
    info = sioux_falls_problem.network.info()
    zero = Vector.copy_of(np.zeros(info.number_of_links))
    free_flow_cost = sioux_falls_problem.cost_fn.compute_link_cost_vector(zero)
    t0 = time.time()
    flow, paths, best_path_cost = load_network(sioux_falls_problem.network,
                               free_flow_cost,
                               sioux_falls_problem.demand)
    warnings.warn(f"Loaded Sioux Falls network in {time.time() - t0} seconds.")
    _flow = flow.to_array()
    assert len(_flow) == info.number_of_links


def test_pittsburgh_network_loading(pittsburgh_problem):
    info = pittsburgh_problem.network.info()
    zero = Vector.copy_of(np.zeros(info.number_of_links))
    free_flow_cost = pittsburgh_problem.cost_fn.compute_link_cost_vector(zero)
    n = 25
    timings = {'free flow': [], 'non-free flow': []}
    for _ in range(n):
        t0 = time.time()
        flow, paths, best_path_cost = load_network(pittsburgh_problem.network,
                                   free_flow_cost,
                                   pittsburgh_problem.demand)
        timings['free flow'].append(time.time() - t0)
        flow_cost = pittsburgh_problem.cost_fn.compute_link_cost_vector(flow)
        t0 = time.time()
        flow, paths, best_path_cost = load_network(pittsburgh_problem.network,
                                   flow_cost,
                                   pittsburgh_problem.demand)
        timings['non-free flow'].append(time.time() - t0)
    fft = timings['free flow']
    nfft = timings['non-free flow']
    warnings.warn(("Loaded Pittsburgh: "
                   f"free flow in {np.mean(fft)}±{np.std(fft)} seconds (best of {n}: {np.min(fft)}); "
                   f"non free flow in {np.mean(nfft)}±{np.std(nfft)} seconds (best of {n}: {np.min(nfft)})")
                  )
