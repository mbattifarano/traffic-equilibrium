# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport stdout, setbuf, printf

from .graph cimport DiGraph
from .trips cimport OrgnDestDemand
from .link_cost cimport LinkCost
from .vector cimport Vector, PointerVector
from .network_loading cimport shortest_paths_assignment
from .igraph_utils cimport vector_len, vector_ptr_len, vector_get, vector_ptr_get, igraph_vector_dot
from .timer cimport now
from .pathdb import PathDB

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

    def __cinit__(self, long int max_iterations, double gap_tolerance, double line_search_tolerance,
                  size_t commit_interval):
        self.max_iterations = max_iterations
        self.gap_tolerance = gap_tolerance
        self.line_search_tolerance = line_search_tolerance
        self.commit_interval = commit_interval

    def save(self, filename):
        with open(f"{filename}.json", "w") as fp:
            json.dump({
                'max_iterations': self.max_iterations,
                'gap_tolerance': self.gap_tolerance,
                'line_search_tolerance': self.line_search_tolerance,
                'commit_interval': self.commit_interval,
            }, fp, indent=2)

    @staticmethod
    def load(filename):
        with open(f"{filename}.json") as fp:
            obj = json.load(fp)
        return FrankWolfeSettings.__new__(FrankWolfeSettings,
                                          obj['max_iterations'],
                                          obj['gap_tolerance'],
                                          obj['line_search_tolerance'],
                                          obj.get('commit_interval', 1000))


cdef class Result:
    def __cinit__(self, Problem problem, FrankWolfeSettings settings,
                  PathDB paths,
                  Vector prev_flow,Vector flow,
                  Vector cost, Vector trip_cost,
                  double gap, long int iterations,
                  double duration):
        self.problem = problem
        self.settings = settings
        self.paths = paths
        self.prev_flow = prev_flow
        self.flow = flow
        self.cost = cost
        self.trip_cost = trip_cost
        self.gap = gap
        self.iterations = iterations
        self.duration = duration

    def improve(self, max_iterations=None, line_search_tolerance=None):
        cdef double tol = self.settings.gap_tolerance
        cdef ConvergenceCriteria metrics = ConvergenceCriteria()
        if max_iterations is not None:
            self.settings.max_iterations = max_iterations
            # if we have set max_iterations ignore gap tolerance (i.e. always run to max iter)
            tol = 0.0
        if line_search_tolerance is not None:
            self.settings.line_search_tolerance = line_search_tolerance
        cdef long int k = 0, max_iter = self.settings.max_iterations
        cdef double gap = self.gap
        cdef Vector next_flow = Vector.zeros(self.problem.network.number_of_links())
        cdef double t0 = now()
        while k < max_iter and gap > tol:
            igraph_vector_update(self.prev_flow.vec, self.flow.vec)
            improve(self.problem, self.settings,
                    self.flow, next_flow, self.cost,
                    self.paths, self.trip_cost, self.problem.demand.volumes,
                    metrics)
            k += 1
            gap = metrics.relative_gap
            if k == 1 or k % self.settings.commit_interval == 0:
                printf("iteration %li; aec: %g; rgap: %g ; timing %g it/s; ... ",
                       k, metrics.average_excess_cost, metrics.relative_gap,
                       k / (now() - t0))
                t1 = now()
                self.paths.commit()
                printf("committed paths in %g seconds.\n", now() - t1)
        t0 = now() - t0
        self.duration += t0
        self.iterations += k
        self.gap = gap
        self.paths.commit()
        printf("improved to average excess cost %g in %li iterations and %g seconds (%g it/s).\n", gap, k, t0, k / t0)
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
                'paths_db': self.paths.name,
            }, fp, indent=2)
        np.savez(os.path.join(dirname, "arrays"),
                 prev_flow=self.prev_flow.to_array(),
                 flow=self.flow.to_array(),
                 cost=self.cost.to_array(),
                 trip_cost=self.trip_cost.to_array(),
        )
        self.settings.save(os.path.join(dirname, 'settings'))
        self.problem.save(dirname)
        return dirname


    @staticmethod
    def load(dirname):
        arrays = np.load(os.path.join(dirname, "arrays.npz"))
        flow = Vector.copy_of(arrays["flow"])
        if "prev_flow" in arrays.keys():
            prev_flow = Vector.copy_of(arrays["prev_flow"])
        else:
            print("Warning: no `prev_flow` found, falling back to `flow`.")
            prev_flow = Vector.copy_of(arrays["flow"])
        cost = Vector.copy_of(arrays["cost"])
        trip_cost = Vector.copy_of(arrays["trip_cost"])
        settings = FrankWolfeSettings.load(os.path.join(dirname, 'settings'))
        problem = Problem.load(dirname)
        with open(os.path.join(dirname, "metadata.json")) as fp:
            obj = json.load(fp)
        paths = PathDB(obj['paths_db'])
        gap = obj['gap']
        iterations = obj['iterations']
        duration = obj['duration']
        return Result.__new__(Result,
            problem,
            settings,
            paths,
            prev_flow, flow,
            cost, trip_cost,
            gap, iterations, duration
        )


def solve(Problem problem, FrankWolfeSettings settings, PathDB paths):
    cdef long int k = 0, n_links = problem.network.number_of_links()
    # set the gap to something larger than the tolerance for the initial iteration
    cdef double t0, t1, t, gap = settings.gap_tolerance + 1
    cdef Vector flow = Vector.zeros(n_links)
    cdef Vector cost = Vector.zeros(n_links)
    cdef Vector next_flow = Vector.zeros(n_links)
    cdef Vector prev_flow = Vector.zeros(n_links)
    cdef Vector volume = Vector.of(problem.demand.volumes.vec)
    cdef Vector trip_cost = Vector.zeros(problem.demand.number_of_trips())
    cdef ConvergenceCriteria metrics = ConvergenceCriteria()
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    cdef double t_iteration = 0.0
    t0 = now()

    printf("Computing initial assignment\n")
    # initial assignment
    shortest_paths_assignment(problem.network,
                              cost,
                              problem.demand,
                              flow,
                              paths,
                              trip_cost
                              )
    printf("Computed initial assignment in %g seconds.\n", now() - t0)
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    while k < settings.max_iterations and gap > settings.gap_tolerance:
        igraph_vector_update(prev_flow.vec, flow.vec)
        improve(problem, settings,
                      flow, next_flow, cost,
                      paths, trip_cost, volume,
                      metrics)
        k += 1
        gap = metrics.relative_gap
        if k == 1 or  k % settings.commit_interval == 0:
            printf("iteration %li; aec: %g; rgap: %g; timing: %g it/s ... ",
                   k,
                   metrics.average_excess_cost,
                   metrics.relative_gap,
                   k / (now() - t0)
                   )
            t1 = now()
            paths.commit()
            printf("committed paths in %g seconds.\n", now() - t1)
    t_iteration = (now() - t0)
    printf("Solved to average excess cost %g and relative gap %g in %li iterations in %g seconds (%g it/s).\n",
           metrics.average_excess_cost, metrics.relative_gap, k, t_iteration, (1.0*k) / t_iteration)
    printf("average timing per iteration %g.\n", t_iteration / k)
    t1 = now()
    paths.commit()
    printf("committed paths in %g seconds.\n", now() - t1)
    return Result(
        problem,
        settings,
        paths,
        prev_flow,
        flow,
        cost,
        trip_cost,
        gap,
        k,
        now() - t0
    )

cdef void improve(Problem problem, FrankWolfeSettings settings,
                  Vector flow, Vector next_flow, Vector cost,
                  PathDB paths,
                  Vector best_path_costs, Vector volume,
                  ConvergenceCriteria metrics,
                  ):
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
    metrics.average_excess_cost = compute_gap(flow, cost, volume, best_path_costs)
    metrics.relative_gap = compute_rgap(flow, cost, volume, best_path_costs)
    #printf(", gap time %gs", t)
    # compute search direction (next_flow is now the search direction)
    igraph_vector_sub(next_flow.vec, flow.vec)
    # update flow and cost
    line_search(problem.cost_fn, flow, cost, next_flow, settings.line_search_tolerance)
    #printf(", line search time %gs\n", t)

cdef double compute_gap(Vector link_flow, Vector link_cost, Vector volume, Vector trip_cost):
    """Average Excess Cost

    See Transportation Network Analysis (v0.85) p 190
    """
    cdef double total_system_travel_time = igraph_vector_dot(
        link_flow.vec, link_cost.vec
    )
    cdef double best_trip_travel_time = igraph_vector_dot(
        volume.vec,
        trip_cost.vec
    )
    cdef double total_volume = igraph_vector_sum(volume.vec)
    #printf("tstt: %g; bptt: %g; volume: %g\n", total_system_travel_time, best_path_travel_time, total_volume)
    return (total_system_travel_time - best_trip_travel_time) / total_volume

cdef double compute_rgap(Vector link_flow, Vector link_cost, Vector volume, Vector trip_cost):
    cdef double total_system_travel_time = igraph_vector_dot(
        link_flow.vec, link_cost.vec
    )
    cdef double best_trip_travel_time = igraph_vector_dot(
        volume.vec,
        trip_cost.vec
    )
    return (total_system_travel_time / best_trip_travel_time) - 1.0

cdef double vector_distance(igraph_vector_t *u, igraph_vector_t *v) nogil:
    cdef double result = 0.0
    cdef long int i, n = vector_len(u)
    for i in range(n):
        result += (vector_get(u, i) - vector_get(v, i))**2
    return result

cdef double line_search(LinkCost cost_fn, Vector flow, Vector cost, Vector direction, double tolerance):
    # cdef double a = 0.0, b = 1.0, sigma = 0.0, alpha
    cdef double sigma = 0.0
    cdef unsigned int k = 0, max_iterations = 256
    cdef igraph_vector_t a, b, c
    igraph_vector_copy(&a, flow.vec)
    igraph_vector_copy(&b, flow.vec)
    igraph_vector_add(&b, direction.vec)
    # igraph_vector_copy(&d, direction.vec)
    while k < max_iterations:
        k += 1
        # reset flow and d
        #igraph_vector_update(flow.vec, &x)
        #igraph_vector_update(&d, direction.vec)
        # compute alpha
        # flow = 0.5 * (a + b)
        igraph_vector_update(flow.vec, &a)
        igraph_vector_add(flow.vec, &b)
        igraph_vector_scale(flow.vec, 0.5)
        # alpha = 0.5 * (a + b)
        # x = flow + alpha * direction
        #igraph_vector_scale(&d, alpha)
        #igraph_vector_add(flow.vec, &d)
        # t = link_cost_fn.compute_link_cost(x, cost)
        cost_fn.compute_link_cost(flow.vec, cost.vec)  # overwrites cost
        # sigma = dot(t, direction)
        sigma = igraph_vector_dot(cost.vec, direction.vec)
        if sigma < 0:
            #a = alpha
            igraph_vector_update(&a, flow.vec)
        else:
            #b = alpha
            igraph_vector_update(&b, flow.vec)
    igraph_vector_destroy(&a)
    igraph_vector_destroy(&b)
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
