# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport stdout, setbuf, printf

from .graph cimport DiGraph
from .trips cimport OrgnDestDemand
from .link_cost cimport LinkCost
from .vector cimport Vector, PointerVector
from .network_loading cimport init_path_vectors, shortest_paths_assignment
from .igraph_utils cimport vector_len, vector_ptr_len, vector_get, vector_ptr_get, igraph_vector_dot
from .timer cimport now
from .path_set cimport PathSet

from .igraph cimport (
    igraph_vector_t, igraph_vector_sub, igraph_vector_add, igraph_vector_scale,
    igraph_vector_copy, igraph_vector_destroy, igraph_vector_update,
    igraph_vector_sum,
    igraph_vector_ptr_t
)

import datetime as dt
import os
import json
import numpy as np


# disable stdout buffer (for debugging)
# setbuf(stdout, NULL)

cdef class Problem:
    cdef readonly DiGraph network
    cdef readonly OrgnDestDemand demand
    cdef readonly LinkCost cost_fn

    def __cinit__(self, DiGraph network, OrgnDestDemand demand, LinkCost cost_fn):
        self.network = network
        self.demand = demand
        self.cost_fn = cost_fn

    def save(self, dirname):
        self.network.save(dirname)
        self.demand.save(dirname)
        self.cost_fn.save(dirname)


cdef class FrankWolfeSettings:
    cdef readonly long int max_iterations
    cdef readonly double gap_tolerance
    cdef readonly double line_search_tolerance

    def __cinit__(self, long int max_iterations, double gap_tolerance, double line_search_tolerance):
        self.max_iterations = max_iterations
        self.gap_tolerance = gap_tolerance
        self.line_search_tolerance = line_search_tolerance

    def save(self, filename):
        with open(f"{filename}.json", "w") as fp:
            json.dump({
                'max_iterations': self.max_iterations,
                'gap_tolerance': self.gap_tolerance,
                'line_search_tolerance': self.line_search_tolerance,
            }, fp, indent=2)


cdef class Result:
    cdef readonly Problem problem
    cdef readonly FrankWolfeSettings settings
    cdef readonly PathSet path_set
    cdef readonly Vector flow
    cdef readonly Vector cost
    cdef readonly double gap
    cdef readonly long int iterations
    cdef readonly double duration

    def __cinit__(self, Problem problem, FrankWolfeSettings settings,
                  PathSet path_set,
                  Vector flow, Vector cost, double gap, long int iterations,
                  double duration):
        self.problem = problem
        self.settings = settings
        self.path_set = path_set
        self.flow = flow
        self.cost = cost
        self.gap = gap
        self.iterations = iterations
        self.duration = duration

    def save(self, name):
        """Save the result to a directory."""
        dirname = os.path.join(name, f"results-{dt.datetime.utcnow().isoformat(timespec='seconds')}")
        os.makedirs(dirname)
        with open(os.path.join(dirname, "metadata.json"), "w") as fp:
            json.dump({
                'gap': self.gap,
                'iterations': self.iterations,
                'duration': self.duration,
            }, fp, indent=2)
        np.save(os.path.join(dirname, 'flow'), self.flow.to_array())
        np.save(os.path.join(dirname, 'cost'), self.cost.to_array())
        self.path_set.write(dirname)
        self.settings.save(os.path.join(dirname, 'settings'))
        self.problem.save(dirname)

    def load(self, dirname):
        """
        flow = Vector.copy_of(np.load(os.path.join(dirname, 'flow')))
        cost = Vector.copy_of(np.load(os.path.join(dirname, 'cost')))
        settings = FrankWolfeSettings.load(os.path.join(dirname, 'settings'))
        path_set = PathSet.load(dirname)
        with open(os.path.join(dirname, "metadata.json")) as fp:
            obj = json.load(fp)
        gap = obj['gap']
        iterations = obj['iterations']
        duration = obj['duration']
        return Result.__new__(Result,
            problem,
            settings,
            path_set,
            flow, cost, gap, iterations, duration
        )
        """
        pass


def solve(Problem problem, FrankWolfeSettings settings):
    cdef long int k = 0, n_links = problem.network.number_of_links()
    # set the gap to something larger than the tolerance for the initial iteration
    cdef double t0, t1, t, gap = settings.gap_tolerance + 1
    cdef Vector flow = Vector.zeros(n_links)
    cdef Vector cost = Vector.zeros(n_links)
    cdef Vector next_flow = Vector.zeros(n_links)
    cdef Vector volume = Vector.of(problem.demand.volumes.vec)
    cdef Vector best_path_costs = Vector.zeros(problem.demand.number_of_trips())
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    cdef PathSet paths = PathSet.__new__(PathSet,
                                         problem.demand.number_of_sources())

    cdef double t_loading = 0.0, t_gap = 0.0, t_line_search = 0.0, t_iteration
    t0 = now()

    printf("Computing initial assignment\n")
    # initial assignment
    shortest_paths_assignment(problem.network,
                              cost,
                              problem.demand,
                              flow,
                              paths,
                              best_path_costs
                              )
    printf("Computed initial assignment in %g seconds.\n", now() - t0)
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    while k < settings.max_iterations and gap > settings.gap_tolerance:
        # all or nothing assignment
        t1 = now()
        shortest_paths_assignment(problem.network,
                                  cost,
                                  problem.demand,
                                  next_flow,
                                  paths,
                                  best_path_costs
                                  )
        t = now() - t1
        t_loading += t
        #printf("%li: loading time %gs", k, t)
        t1 = now()
        # compute gap
        gap = compute_gap(flow, cost, volume, best_path_costs)
        t = now() - t1
        t_gap += t
        #printf(", gap time %gs", t)
        t1 = now()
        # compute search direction (next_flow is now the search direction)
        igraph_vector_sub(next_flow.vec, flow.vec)
        # update flow and cost
        line_search(problem.cost_fn, flow, cost, next_flow, settings.line_search_tolerance)
        t = now() - t1
        t_line_search += t
        #printf(", line search time %gs\n", t)
        # printf("%li: gap: %g\n", k, gap)
        k += 1
    t_gap /= k
    t_loading /= k
    t_line_search /= k
    t_iteration = t_gap + t_loading + t_line_search
    printf("path set stores %li paths using %li bytes.\n", paths.number_of_paths(), paths.memory_usage())
    printf("average timings per iteration %g:\n\tnetwork loading: %g (%g%%)\n\tgap: %g (%g%%)\n\tline search: %g (%g%%)\n",
           t_iteration,
           t_loading, 100.0 * t_loading / t_iteration,
           t_gap, 100.0 * t_gap / t_iteration,
           t_line_search, 100.0 * t_line_search / t_iteration,
           )
    return Result(
        problem,
        settings,
        paths,
        flow,
        cost,
        gap,
        k,
        now() - t0
    )

# TODO not correct fix.
cdef double compute_gap(Vector link_flow, Vector link_cost, Vector volume, Vector best_path_cost):
    """Average Excess Cost

    See Transportation Network Analysis (v0.85) p 190
    """
    cdef double total_system_travel_time = igraph_vector_dot(
        link_flow.vec, link_cost.vec
    )
    cdef double best_path_travel_time = igraph_vector_dot(
        volume.vec,
        best_path_cost.vec
    )
    cdef double total_volume = igraph_vector_sum(volume.vec)
    #printf("tstt: %g; bptt: %g; volume: %g\n", total_system_travel_time, best_path_travel_time, total_volume)
    return (total_system_travel_time - best_path_travel_time) / total_volume


cdef double line_search(LinkCost cost_fn, Vector flow, Vector cost, Vector direction, double tolerance):
    cdef double a = 0.0, b = 1.0, sigma = 0.0, alpha
    cdef igraph_vector_t x, d
    igraph_vector_copy(&x, flow.vec)
    igraph_vector_copy(&d, direction.vec)
    while b - a > tolerance or sigma > 0:
        # reset flow and d
        igraph_vector_update(flow.vec, &x)
        igraph_vector_update(&d, direction.vec)
        # compute alpha
        alpha = 0.5 * (a + b)
        # x = flow + alpha * direction
        igraph_vector_scale(&d, alpha)
        igraph_vector_add(flow.vec, &d)
        # t = link_cost_fn.compute_link_cost(x, cost)
        cost_fn.compute_link_cost(flow.vec, cost.vec)  # overwrites cost
        # sigma = dot(t, direction)
        sigma = igraph_vector_dot(cost.vec, direction.vec)
        if sigma < 0:
            a = alpha
        else:
            b = alpha
    igraph_vector_destroy(&x)
    igraph_vector_destroy(&d)
    return -sigma



cdef double line_search_(LinkCost cost_fn, Vector flow, Vector cost, Vector direction, double tolerance):
    cdef double alpha, beta, alpha_pre = 0.0, delta = 1.0, sigma = 0.0, a = 0.0, b = 1.0
    cdef double drift = 1.0
    while (b - a) > tolerance or sigma > 0:
        alpha = 0.5 * (a + b)
        # in place update direction
        beta = (alpha - alpha_pre) / delta
        drift *= beta  # accumulate all multiplies
        igraph_vector_scale(direction.vec, beta)
        delta = alpha - alpha_pre
        alpha_pre = alpha
        # in place update flow
        # x = flow + alpha * direction
        igraph_vector_add(flow.vec, direction.vec)
        # in place update cost
        # t = cost(flow)
        cost_fn.compute_link_cost(flow.vec, cost.vec)
        # compute sigma (re-scale direction to reverse inplace updates)
        # sigma = t . direction
        sigma = igraph_vector_dot(cost.vec, direction.vec) / drift
        if sigma < 0:
            a = alpha
        else:
            b = alpha
    return -sigma
