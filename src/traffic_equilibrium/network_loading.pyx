# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport stdout, setbuf, printf
from libc.stdlib cimport free, malloc
from .path_set cimport PathSet
from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand
from .timer cimport now
from .dang cimport dang_t


from .igraph cimport (
    igraph_integer_t, igraph_get_shortest_paths_dijkstra, igraph_neimode_t,
    igraph_vector_t, igraph_vector_ptr_t, igraph_vss_vector, igraph_real_t,
    igraph_matrix_t, igraph_matrix_e_ptr, igraph_matrix_init, igraph_matrix_e,
    igraph_matrix_destroy, igraph_vector_destroy, igraph_finally_func_t,
    igraph_vs_t, igraph_vector_init, igraph_vector_ptr_set_item_destructor,
    igraph_vector_ptr_view, igraph_vector_ptr_destroy_all, igraph_t,
    igraph_vector_long_t, igraph_vector_view, igraph_vector_e_ptr,
    igraph_dqueue_t, igraph_dqueue_init, igraph_dqueue_push, igraph_dqueue_pop,
    igraph_dqueue_pop_back, igraph_dqueue_head, igraph_dqueue_back, igraph_dqueue_empty
)
from .igraph_utils cimport (
    vector_get, vector_ptr_get, vector_len, vector_set, vector_ptr_set
)
from cython.parallel cimport prange, threadid
from openmp cimport omp_get_max_threads


# disable stdout buffer (for debugging)
# setbuf(stdout, NULL)

cdef extern from "shortest_paths.c":
    int get_shortest_paths_bellman_ford(
            igraph_t *graph,
            dang_t* path_trie,
            igraph_real_t *path_costs, # cost of each path
            long int source,
            igraph_vs_t to,
            igraph_vector_t *weights,
            igraph_real_t *link_flow,
            igraph_real_t *volumes
    ) nogil

    int dqueue_push_front(
        igraph_dqueue_t *q,
        igraph_real_t elem
    ) nogil

cpdef test_dqueue_push_front():
    cdef igraph_dqueue_t q
    igraph_dqueue_init(&q, 5)
    dqueue_push_front(&q, 1.0)
    #print(f"head is {igraph_dqueue_head(&q)} == 1.0")
    assert igraph_dqueue_pop(&q) == 1.0
    igraph_dqueue_push(&q, 2.0)
    #print(f"head is {igraph_dqueue_head(&q)} == 2.0")
    dqueue_push_front(&q, 3.0)
    #print(f"head is {igraph_dqueue_head(&q)} == 3.0")
    #print(f"tail is {igraph_dqueue_back(&q)} == 2.0")
    assert igraph_dqueue_head(&q) == 3.0
    assert igraph_dqueue_back(&q) == 2.0
    dqueue_push_front(&q, 4.0)
    #print(f"head is {igraph_dqueue_head(&q)} == 4.0")
    #print(f"tail is {igraph_dqueue_back(&q)} == 2.0")
    assert igraph_dqueue_head(&q) == 4.0
    assert igraph_dqueue_back(&q) == 2.0
    #print("popping all")
    #print(f"head is {igraph_dqueue_head(&q)} == 4.0")
    assert igraph_dqueue_pop(&q) == 4.0
    #print(f"head is {igraph_dqueue_head(&q)} == 3.0")
    assert igraph_dqueue_pop(&q) == 3.0
    #print(f"head is {igraph_dqueue_head(&q)} == 2.0")
    assert igraph_dqueue_pop(&q) == 2.0
    assert igraph_dqueue_empty(&q)


cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PathSet paths,
                                    Vector best_path_cost
                                    ) nogil:
    cdef:
        long int i, j, k, eid, first_trip_index
        long int number_of_sources = demand.number_of_sources()
        long int number_of_targets
        long int count = 0
        long int source
        long int index
        igraph_real_t volume
        igraph_real_t *volumes_for_source
        igraph_real_t *costs_for_source
        igraph_vector_t* targets_vec
        igraph_vs_t targets_vs
        igraph_vector_t* path
        int n_threads = omp_get_max_threads()
        int thread_id
        igraph_matrix_t _flows
        igraph_real_t* _flow
        igraph_real_t _flow_at
    igraph_matrix_init(&_flows, network.number_of_links(), n_threads)

    #printf("Running shortest paths...\n")
    for i in prange(number_of_sources, num_threads=n_threads, schedule='dynamic', chunksize=12):
    #for i in range(number_of_sources):
        #printf("%li getting source... ", i)
        source = <igraph_integer_t> vector_get(demand.sources.vec, i)
        ##printf("%li getting targets... ", i)
        targets_vec = <igraph_vector_t*> vector_ptr_get(demand.targets.vec, i)
        number_of_targets = vector_len(targets_vec)
        if number_of_targets > 0:
            ##printf("%li getting trip indices...", i)
            trip_indices = <igraph_vector_t*> vector_ptr_get(demand.trip_index.vec, i)
            first_trip_index = <igraph_integer_t> vector_get(trip_indices, 0)
            volumes_for_source = igraph_vector_e_ptr(demand.volumes.vec, first_trip_index)
            costs_for_source = igraph_vector_e_ptr(best_path_cost.vec, first_trip_index)
            ##printf("%li getting paths for source...", i)
            # get the linear index of the first trip
            targets_vs = igraph_vss_vector(targets_vec)
            thread_id = threadid()
            # populate paths with shortest paths vectors (of edge ids)
            _flow = <igraph_real_t*> igraph_matrix_e_ptr(&_flows, 0, thread_id)
            #printf("%li getting shortest paths... ", i)
            get_shortest_paths_bellman_ford(
                network.graph,
                paths.get_trie(i),
                costs_for_source,
                source,
                targets_vs,
                cost.vec,
                _flow,
                volumes_for_source
            )
            #printf("got shortest paths for %li... ", i)
    # sum the thread local link flows into the final link flow
    for eid in range(vector_len(flow.vec)):
        _flow_at = 0.0
        for j in range(n_threads):
            _flow_at += igraph_matrix_e(&_flows, eid, j)
        vector_set(flow.vec, eid, _flow_at)
    igraph_matrix_destroy(&_flows)
    #printf("\n")


# TODO: remove unused
cdef PointerVector init_path_vectors(OrgnDestDemand demand):
    cdef long int i, j, n_targets, n_sources = demand.number_of_sources()
    cdef PointerVector paths = PointerVector.nulls(n_sources)
    cdef PointerVector paths_for_source
    igraph_vector_ptr_set_item_destructor(paths.vec, <igraph_finally_func_t*> igraph_vector_ptr_destroy_all)
    cdef Vector path
    for i in range(n_sources):
        n_targets = demand.number_of_targets(i)
        paths_for_source = PointerVector.nulls(n_targets, owner=False)
        vector_ptr_set(paths.vec, i, paths_for_source.vec)
        igraph_vector_ptr_set_item_destructor(paths_for_source.vec, <igraph_finally_func_t*> igraph_vector_destroy)
        for j in range(n_targets):
            path = Vector.zeros(0, owner=False)
            vector_ptr_set(paths_for_source.vec, j, path.vec)
    return paths


def load_network(DiGraph network, Vector cost, OrgnDestDemand demand):
    cdef Vector flow = Vector.zeros(network.number_of_links())
    cdef long int i, number_of_sources = demand.number_of_sources()
    cdef PathSet paths = PathSet.__new__(PathSet, number_of_sources)
    cdef Vector best_path_cost = Vector.zeros(demand.number_of_trips())
    shortest_paths_assignment(network, cost, demand, flow, paths, best_path_cost)
    return flow, paths, best_path_cost
