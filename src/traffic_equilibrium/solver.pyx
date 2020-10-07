# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from .graph cimport DiGraph
from .trips cimport OrgnDestDemand
from .link_cost cimport LinkCost
from .vector cimport Vector, PointerVector
from .network_loading cimport init_path_vectors, shortest_paths_assignment
from .igraph_utils cimport vector_len, vector_ptr_len, vector_get, vector_ptr_get, igraph_vector_dot
from .timer cimport now

from .igraph cimport igraph_vector_t, igraph_vector_sub, igraph_vector_add, igraph_vector_scale


cdef class Problem:
    cdef readonly DiGraph network
    cdef readonly OrgnDestDemand demand
    cdef readonly LinkCost cost_fn

    def __cinit__(self, DiGraph network, OrgnDestDemand demand, LinkCost cost_fn):
        self.network = network
        self.demand = demand
        self.cost_fn = cost_fn


cdef class FrankWolfeSettings:
    cdef readonly long int max_iterations
    cdef readonly double gap_tolerance
    cdef readonly double line_search_tolerance

    def __cinit__(self, long int max_iterations, double gap_tolerance, double line_search_tolerance):
        self.max_iterations = max_iterations
        self.gap_tolerance = gap_tolerance
        self.line_search_tolerance = line_search_tolerance


cdef class Result:
    cdef readonly Problem problem
    cdef readonly FrankWolfeSettings settings
    cdef readonly Vector flow
    cdef readonly Vector cost
    cdef readonly double gap
    cdef readonly long int iterations
    cdef readonly double duration

    def __cinit__(self, Problem problem, FrankWolfeSettings settings,
                  Vector flow, Vector cost, double gap, long int iterations,
                  double duration):
        self.problem = problem
        self.settings = settings
        self.flow = flow
        self.cost = cost
        self.gap = gap
        self.iterations = iterations
        self.duration = duration


def solve(Problem problem, FrankWolfeSettings settings):
    cdef long int k = 0, n_links = problem.network.number_of_links()
    # set the gap to something larger than the tolerance for the initial iteration
    cdef double gap = settings.gap_tolerance + 1.0
    cdef Vector flow = Vector.zeros(n_links)
    cdef Vector cost = Vector.zeros(n_links)
    cdef Vector next_flow = Vector.zeros(n_links)
    cdef Vector path_costs = Vector.zeros(problem.demand.number_of_trips())
    cdef Vector volume = Vector.of(problem.demand.volumes.vec)
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    cdef PointerVector paths = init_path_vectors(problem.demand)
    cdef double t0 = now()

    # initial assignment
    shortest_paths_assignment(problem.network,
                              cost,
                              problem.demand,
                              flow,
                              paths
                              )
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    while k < settings.max_iterations and gap > settings.gap_tolerance:
        # all or nothing assignment
        shortest_paths_assignment(problem.network,
                                  cost,
                                  problem.demand,
                                  next_flow,
                                  paths
                                  )
        # compute search direction (next_flow is now the search direction)
        igraph_vector_sub(next_flow.vec, flow.vec)
        # update flow and cost
        line_search(problem.cost_fn, flow, cost, next_flow, settings.line_search_tolerance)
        # compute gap
        gap = compute_gap(flow, cost, volume, paths)
        k += 1
    return Result(
        problem,
        settings,
        flow,
        cost,
        gap,
        k,
        now() - t0
    )

cdef double compute_gap(Vector link_flow, Vector link_cost, Vector volume, PointerVector paths):
    cdef double total_system_travel_time = igraph_vector_dot(
        link_flow.vec, link_cost.vec
    )
    cdef long int i, j, eid, n = vector_ptr_len(paths.vec)
    cdef igraph_vector_t* path
    cdef double best_path_travel_time = 0.0
    cdef double _cost
    for i in range(n):
        path = <igraph_vector_t*> vector_ptr_get(paths.vec, i)
        _cost = 0.0
        for j in range(vector_len(path)):
            eid = <long int> vector_get(path, j)
            _cost += vector_get(link_cost.vec, eid)
        best_path_travel_time += (_cost * vector_get(volume.vec, i))
    return (total_system_travel_time / best_path_travel_time) - 1.0


cdef double line_search(LinkCost cost_fn, Vector flow, Vector cost, Vector direction, double tolerance):
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
