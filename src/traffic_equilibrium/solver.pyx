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

DEF BYTES_PER_GIBIBYTE = 1073742000.0
# disable stdout buffer (for debugging)
# setbuf(stdout, NULL)

cpdef enum SolverMode:
    USER = 1
    SYSTEM = 2

cdef class Problem:

    def __cinit__(self, DiGraph network, OrgnDestDemand demand, LinkCost cost_fn):
        self.network = network
        self.demand = demand
        self.cost_fn = cost_fn

    def save(self, dirname):
        self.network.save(dirname)
        self.demand.save(dirname)
        self.cost_fn.save(dirname)

    @staticmethod
    def load(dirname):
        return Problem.__new__(Problem,
                               DiGraph.load(dirname),
                               OrgnDestDemand.load(dirname),
                               LinkCost.load(dirname)
                               )


cdef class FrankWolfeSettings:

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

    @staticmethod
    def load(filename):
        with open(f"{filename}.json") as fp:
            obj = json.load(fp)
        return FrankWolfeSettings.__new__(FrankWolfeSettings,
                                          obj['max_iterations'],
                                          obj['gap_tolerance'],
                                          obj['line_search_tolerance'])


cdef class Result:

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

    def improve(self):
        cdef long int k = 0, max_iter = self.settings.max_iterations
        cdef double gap = self.gap, tol = self.settings.gap_tolerance
        cdef Vector next_flow = Vector.zeros(self.problem.network.number_of_links())
        cdef Vector best_path_costs = Vector.zeros(self.problem.demand.number_of_trips())
        cdef double t0 = now()
        while k < max_iter and gap > tol:
            gap = improve(self.problem, self.settings,
                          self.flow, next_flow, self.cost,
                          self.path_set, best_path_costs, self.problem.demand.volumes)
            k += 1
            # printf("- {iteration: %li, gap: %g}\n", k, gap)
        t0 = now() - t0
        self.duration += t0
        self.iterations += k
        self.gap = gap
        printf("solved to average excess cost %g in %li iterations.\n", gap, k)
        printf("path set stores %li paths using %g GB.\n", self.path_set.number_of_paths(), self.path_set.memory_usage() / 1000000000.0)
        printf("average timing per iteration %g.\n", k / self.duration)
        return self

    def save(self, name, timestamp=None):
        """Save the result to a directory."""
        if timestamp is None:
            timestamp = dt.datetime.utcnow().isoformat(timespec='seconds')
        dirname = os.path.join(name, f"results-{timestamp}")
        os.makedirs(dirname)
        print(f"Saving to {dirname}")
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
        return dirname


    @staticmethod
    def load(dirname, with_pathset=True):
        flow = Vector.copy_of(np.load(os.path.join(dirname, 'flow.npy')))
        cost = Vector.copy_of(np.load(os.path.join(dirname, 'cost.npy')))
        settings = FrankWolfeSettings.load(os.path.join(dirname, 'settings'))
        problem = Problem.load(dirname)
        if with_pathset:
            path_set = PathSet.load(dirname)
        else:
            n_sources, _ = problem.demand.info()
            path_set = PathSet(n_sources)
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

    cdef double t_iteration = 0.0
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
        gap = improve(problem, settings,
                      flow, next_flow, cost,
                      paths, best_path_costs, volume)
        k += 1
        # printf("- {iteration: %li, gap: %g}\n", k, gap)
    t_iteration = (now() - t0) / k
    printf("solved to average excess cost %g in %li iterations.\n", gap, k)
    printf("path set stores %li paths using %g GiB.\n", paths.number_of_paths(), paths.memory_usage() / BYTES_PER_GIBIBYTE)
    printf("average timing per iteration %g.\n", t_iteration)
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

cdef inline double improve(Problem problem, FrankWolfeSettings settings,
                         Vector flow, Vector next_flow, Vector cost,
                         PathSet paths,
                         Vector best_path_costs, Vector volume):
    # all or nothing assignment
    shortest_paths_assignment(problem.network,
                              cost,
                              problem.demand,
                              next_flow,
                              paths,
                              best_path_costs
                              )
    #printf("%li: loading time %gs", k, t)
    # compute gap
    gap = compute_gap(flow, cost, volume, best_path_costs)
    #printf(", gap time %gs", t)
    # compute search direction (next_flow is now the search direction)
    igraph_vector_sub(next_flow.vec, flow.vec)
    # update flow and cost
    line_search(problem.cost_fn, flow, cost, next_flow, settings.line_search_tolerance)
    #printf(", line search time %gs\n", t)
    return gap

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
