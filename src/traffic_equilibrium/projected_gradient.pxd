from .igraph cimport igraph_vector_t, igraph_vector_ptr_t
from .solver cimport Problem
from .vector cimport Vector
from .link_cost cimport LinkCost
from .pathdb cimport PathDB
from .leveldb cimport leveldb_writebatch_t

cdef void igraph_vector_e_add(igraph_vector_t *v, long int at, double plus) nogil

cdef struct path_t:
    double flow
    double cost
    igraph_vector_t link_ids
    bint usable

cdef int path_init(path_t *path) nogil
cdef int path_destroy(path_t *path) nogil
cdef bint path_has_flow(path_t *path) nogil
cdef bint path_is_congruent(path_t *path, igraph_vector_t *link_ids) nogil
cdef size_t path_get_link_id(path_t *path, size_t i) nogil
cdef int path_set_links(path_t* path, igraph_vector_t * link_ids) nogil
cdef int path_to_link_flow(path_t* path, igraph_vector_t* link_flow) nogil
cdef int path_update_link_flow(path_t* path, double new_flow, igraph_vector_t* link_flow) nogil
cdef int path_update_cost(path_t* path, igraph_vector_t* link_cost) nogil
cdef void path_to_leveldb(path_t *path, size_t trip_id, leveldb_writebatch_t *writer) nogil

cdef struct trip_paths_t:
    size_t trip_id
    size_t source
    size_t target
    double demand
    size_t number_unused
    igraph_vector_ptr_t * paths

cdef int trip_paths_init(trip_paths_t* trip_paths, size_t trip_id,
                         size_t source, size_t target, double demand) nogil
cdef int trip_paths_destroy(trip_paths_t* trip_paths) nogil
cdef size_t trip_paths_number_of_paths(trip_paths_t* trip_paths) nogil
cdef path_t* trip_paths_get_path(trip_paths_t *trip_paths, size_t i) nogil
cdef bint trip_paths_add(trip_paths_t * trip_paths, igraph_vector_t * link_ids,
                        double cost) nogil
cdef size_t trip_paths_remove_unused(trip_paths_t* trip_paths, leveldb_writebatch_t *writer) nogil
cdef void trip_paths_to_leveldb(trip_paths_t *trip_paths,
                                leveldb_writebatch_t *writer)
cdef int trip_paths_to_link_flow(trip_paths_t * trip_paths,
                                 igraph_vector_t * link_flow) nogil
cdef int trip_paths_update_path_cost(trip_paths_t * trip_paths,
                                     igraph_vector_t * link_flow) nogil
cdef double trip_paths_fix_feasibility(trip_paths_t *trip_paths,
                                       double min_path_flow,
                                       igraph_vector_t *link_flow) nogil
cdef void trip_paths_display(trip_paths_t *trip_paths, int level) nogil
cdef double trip_paths_search_direction(trip_paths_t* trip_paths, igraph_vector_t* direction) nogil
cdef double trip_paths_line_search(trip_paths_t* trip_paths,
                                   LinkCost cost_fn,
                                   igraph_vector_t* direction,
                                   double max_step_size,
                                   igraph_vector_t* link_flow,
                                   igraph_vector_t* link_cost,
                                   double tolerance)

cdef struct costs_t:
    double total_flow
    double total_cost
    double total_shortest_paths_cost
    double maximum_excess_cost

cdef void costs_reset(costs_t *costs) nogil
cdef void costs_update(costs_t *costs,
                       double total_cost, double total_flow,
                       double max_cost, double min_cost) nogil

cdef struct metrics_t:
    double relative_gap
    double maximum_excess_cost
    double average_excess_cost
    size_t iterations

cdef bint is_converged(metrics_t* tolerance, metrics_t* values) nogil

cdef class PathSet:
    cdef trip_paths_t* trip_paths
    cdef readonly size_t number_of_trips
    cdef readonly PathDB path_db

    cdef size_t number_of_paths(self) nogil
    cdef trip_paths_t* get_trip_paths(self, size_t i) nogil
    cdef leveldb_writebatch_t* get_writer(self, size_t i) nogil
    cdef int to_link_flow(self, igraph_vector_t * link_flow) nogil
    cdef void update_path_cost(self, igraph_vector_t *link_cost)
    cdef bint is_feasible(self, igraph_vector_t* volumes)
    cdef void display(self)
    cdef void save_paths(self)
    cdef void commit(self) nogil

cdef size_t update_path_flow(Problem problem, PathSet path_set,
                             igraph_vector_t *link_cost,
                             igraph_vector_t *link_flow,
                             igraph_vector_t *direction,
                             igraph_vector_t *path_links,
                             double tolerance,
                             metrics_t *metrics,
                             leveldb_writebatch_t *writer)

cdef void metrics_clear(metrics_t * metrics) nogil
cdef bint is_converged(metrics_t *tolerance, metrics_t *values) nogil

cdef class ProjectedGradientSettings:
    cdef metrics_t tolerance
    cdef double line_search_tolerance
    cdef size_t report_every

cdef class ProjectedGradientResult:
    cdef metrics_t metrics
    cdef metrics_t best_metrics
    cdef readonly Problem problem
    cdef readonly PathSet path_set
    cdef readonly PathDB paths
    cdef readonly ProjectedGradientSettings settings
    cdef readonly Vector flow

    cdef void print_report(self, size_t n_paths, double t0)
    cdef void update_path_costs(self, igraph_vector_t* link_flow,
                                igraph_vector_t* link_cost)
    cdef bint is_feasible(self)
    cpdef void solve_via_projected_gradient(self)

