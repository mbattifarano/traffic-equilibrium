# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport stdout, setbuf, printf

from .graph cimport DiGraph
from .trips cimport OrgnDestDemand
from .link_cost cimport LinkCost
from .vector cimport Vector, PointerVector
from .network_loading cimport init_path_vectors, shortest_paths_assignment
from .igraph_utils cimport vector_len, vector_ptr_len, vector_get, vector_ptr_get, igraph_vector_dot
from .timer cimport now

from .igraph cimport (
    igraph_vector_t, igraph_vector_sub, igraph_vector_add, igraph_vector_scale,
    igraph_vector_copy, igraph_vector_destroy, igraph_vector_update,
    igraph_vector_ptr_t
)

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
    cdef double t0, t1, gap = settings.gap_tolerance + 1
    cdef Vector flow = Vector.zeros(n_links)
    cdef Vector cost = Vector.zeros(n_links)
    cdef Vector next_flow = Vector.zeros(n_links)
    cdef Vector volume = Vector.of(problem.demand.volumes.vec)
    problem.cost_fn.compute_link_cost(flow.vec, cost.vec)
    cdef PointerVector paths = init_path_vectors(problem.demand)

    cdef double t_loading = 0.0, t_gap = 0.0, t_line_search = 0.0, t_iteration
    t0 = now()

    printf("Computing initial assignment\n")
    # initial assignment
    shortest_paths_assignment(problem.network,
                              cost,
                              problem.demand,
                              flow,
                              paths
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
                                  paths
                                  )
        t_loading += now() - t1
        t1 = now()
        # compute gap
        gap = compute_gap(flow, cost, volume, paths)
        # printf("%li: gap = %g;", k, gap)
        t_gap += now() - t1
        t1 = now()
        # compute search direction (next_flow is now the search direction)
        igraph_vector_sub(next_flow.vec, flow.vec)
        # update flow and cost
        line_search(problem.cost_fn, flow, cost, next_flow, settings.line_search_tolerance)
        t_line_search += now() - t1
        k += 1
    t_gap /= k
    t_loading /= k
    t_line_search /= k
    t_iteration = t_gap + t_loading + t_line_search
    printf("average timings per iteration %g:\n\tnetwork loading: %g (%g%%)\n\tgap: %g (%g%%)\n\tline search: %g (%g%%)\n",
           t_iteration,
           t_loading, 100.0 * t_loading / t_iteration,
           t_gap, 100.0 * t_gap / t_iteration,
           t_line_search, 100.0 * t_line_search / t_iteration,
           )
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
    """Average Excess Cost 
    
    see Transportation Network Analysis (v0.85) p 190
    """
    cdef double total_system_travel_time = igraph_vector_dot(
        link_flow.vec, link_cost.vec
    )
    cdef long int i, j, k, l, eid, n_targets, n_sources = vector_ptr_len(paths.vec)
    cdef igraph_vector_ptr_t* paths_for_source
    cdef igraph_vector_t* path
    cdef double best_path_travel_time = 0.0
    cdef double _cost, _volume, total_volume = 0.0
    # compute the cost of each path
    k = 0
    for i in range(n_sources):
        paths_for_source = <igraph_vector_ptr_t*> vector_ptr_get(paths.vec, i)
        n_targets = vector_ptr_len(paths_for_source)
        for j in range(n_targets):
            path = <igraph_vector_t*> vector_ptr_get(paths_for_source, j)
            _cost = 0.0
            _volume = vector_get(volume.vec, k)
            total_volume += _volume
            for l in range(vector_len(path)):
                eid = <long int> vector_get(path, l)
                _cost += vector_get(link_cost.vec, eid)
            best_path_travel_time += (_cost * _volume)
            k += 1
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
