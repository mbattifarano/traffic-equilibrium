# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport stdout, setbuf, printf
from libc.stdlib cimport free, malloc
from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand
from .timer cimport now

from .igraph cimport (
    igraph_integer_t, igraph_get_shortest_paths_dijkstra, igraph_neimode_t,
    igraph_vector_t, igraph_vector_ptr_t, igraph_vss_vector, igraph_real_t,
    igraph_matrix_t, igraph_matrix_e_ptr, igraph_matrix_init, igraph_matrix_e,
    igraph_matrix_destroy, igraph_vector_destroy, igraph_finally_func_t,
    igraph_vs_t, igraph_vector_init, igraph_vector_ptr_set_item_destructor,
    igraph_vector_ptr_view, igraph_vector_ptr_destroy_all, igraph_t
)
from .igraph_utils cimport (
    vector_get, vector_ptr_get, vector_len, vector_set, vector_ptr_set
)
from cython.parallel cimport prange, threadid
from openmp cimport omp_get_max_threads


# disable stdout buffer (for debugging)
#setbuf(stdout, NULL)

cdef extern from "shortest_paths.c":
    int igraph_get_shortest_paths_bellman_ford(
            igraph_t *graph,
            igraph_vector_ptr_t *edges, # paths from from to each of to
            igraph_vector_t *path_costs, # cost of each path
            igraph_integer_t from_,
            igraph_vs_t to,
            igraph_vector_t *weights,
            igraph_neimode_t mode
    ) nogil

cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PointerVector paths,
                                    ) nogil:
    #printf("shortest_path_assignment:...")
    cdef:
        long int i, j, k, eid, first_trip_index
        long int number_of_sources = demand.number_of_sources()
        long int number_of_targets
        long int count = 0
        long int source
        long int index
        igraph_real_t volume
        igraph_vector_t* targets_vec
        igraph_vs_t targets_vs
        igraph_vector_ptr_t* paths_for_source
        igraph_vector_t* path
        int n_threads = omp_get_max_threads()
        int thread_id
        igraph_matrix_t _flows
        igraph_real_t* _flow
        igraph_real_t _flow_at
        bint use_label_correcting = False
    igraph_matrix_init(&_flows, network.number_of_links(), n_threads)

    for i in prange(number_of_sources, num_threads=n_threads, schedule='guided', chunksize=32):
    #for i in range(number_of_sources):
        #printf("%li getting source...", i)
        source = <igraph_integer_t> vector_get(demand.sources.vec, i)
        #printf("%li getting targets...", i)
        targets_vec = <igraph_vector_t*> vector_ptr_get(demand.targets.vec, i)
        if vector_len(targets_vec) > 0:
            number_of_targets = vector_len(targets_vec)
            #printf("%li getting trip indices...", i)
            trip_indices = <igraph_vector_t*> vector_ptr_get(demand.trip_index.vec, i)
            #printf("%li getting paths for source...", i)
            paths_for_source = <igraph_vector_ptr_t*> vector_ptr_get(paths.vec, i)
            # get the linear index of the first trip
            targets_vs = igraph_vss_vector(targets_vec)
            thread_id = threadid()
            # populate paths with shortest paths vectors (of edge ids)
            #printf("%li getting shortest paths...", i)
            get_shortest_paths(
                network.graph,
                paths_for_source,
                source,
                targets_vs,
                cost.vec,
                use_label_correcting
            )
            # iterate through shortest paths and load the links with volume
            # link flow is thread local
            #printf("%li computing link flow...", i)
            for j in range(number_of_targets):
                path = <igraph_vector_t*> vector_ptr_get(paths_for_source, j)
                # TODO: (thread safely) add the path to the path set
                index = <long int> vector_get(trip_indices, j)
                volume = vector_get(demand.volumes.vec, index)
                # matrices are stored column-wise, grab a pointer to the first
                # element of the column for this thread
                _flow = <igraph_real_t*> igraph_matrix_e_ptr(&_flows, 0, thread_id)
                #printf("Found path of length %li: ", vector_len(path))
                for k in range(vector_len(path)):
                    eid = <long int> vector_get(path, k)
                    #printf("%li ", eid)
                    _flow[eid] += volume
                #printf("\n")
            #printf("%li done", i)
    # sum the thread local link flows into the final link flow
    for eid in range(vector_len(flow.vec)):
        _flow_at = 0.0
        for j in range(n_threads):
            _flow_at += igraph_matrix_e(&_flows, eid, j)
        vector_set(flow.vec, eid, _flow_at)
    igraph_matrix_destroy(&_flows)
    #printf("...done\n")

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
    cdef PointerVector paths = init_path_vectors(demand)
    shortest_paths_assignment(network, cost, demand, flow, paths)
    return flow, paths

cdef inline void get_shortest_paths(
        igraph_t *graph,
        igraph_vector_ptr_t *paths_for_source,
        igraph_integer_t source,
        igraph_vs_t targets,
        igraph_vector_t *cost,
        bint label_correcting
) nogil:
    cdef igraph_neimode_t mode = igraph_neimode_t.IGRAPH_OUT
    if label_correcting is True:
        igraph_get_shortest_paths_bellman_ford(
            graph,
            paths_for_source,
            NULL,  # path costs
            source,
            targets,
            cost,
            mode
        )
    else:
        igraph_get_shortest_paths_dijkstra(
            graph,
            NULL,  # nodes
            paths_for_source,
            source,
            targets,
            cost,
            mode,
            NULL,  # predecessors
            NULL,  # inbound_links
        )
