# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdlib cimport free, malloc
from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand

from .igraph cimport (
    igraph_integer_t, igraph_get_shortest_paths_dijkstra, igraph_neimode_t,
    igraph_vector_t, igraph_vector_ptr_t, igraph_vss_vector, igraph_real_t,
    igraph_matrix_t, igraph_matrix_e_ptr, igraph_matrix_init, igraph_matrix_e,
    igraph_matrix_destroy, igraph_vector_destroy, igraph_finally_func_t,
    igraph_vs_t, igraph_vector_init, igraph_vector_ptr_set_item_destructor,
    igraph_vector_ptr_view
)
from .igraph_utils cimport (
    vector_get, vector_ptr_get, vector_len, vector_set, vector_ptr_set
)
from cython.parallel cimport prange, threadid
from openmp cimport omp_get_num_threads

cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PointerVector paths,
                                    ) nogil:
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
        igraph_vector_ptr_t paths_for_source
        igraph_vector_t* path
        int n_threads = omp_get_num_threads()
        int thread_id
        igraph_matrix_t _flows
        igraph_real_t* _flow
        igraph_real_t _flow_at
    igraph_matrix_init(&_flows, network.number_of_links(), n_threads)

    for i in prange(number_of_sources, schedule='static', chunksize=64):
    #for i in range(number_of_sources):
        source = <igraph_integer_t> vector_get(demand.sources.vec, i)
        targets_vec = <igraph_vector_t*> vector_ptr_get(demand.targets.vec, i)
        if vector_len(targets_vec) > 0:
            number_of_targets = vector_len(targets_vec)
            trip_indices = <igraph_vector_t*> vector_ptr_get(demand.trip_index.vec, i)
            # get the linear index of the first trip
            first_trip_index = <long int> vector_get(trip_indices, 0)
            targets_vs = igraph_vss_vector(targets_vec)
            thread_id = threadid()
            igraph_vector_ptr_view(&paths_for_source, paths.vec.stor_begin + first_trip_index, number_of_targets)
            # populate paths with shortest paths vectors (of edge ids)
            igraph_get_shortest_paths_dijkstra(
                network.graph,
                NULL,  # nodes
                &paths_for_source,
                source,
                targets_vs,
                cost.vec,
                igraph_neimode_t.IGRAPH_OUT,
                NULL,  # predecessors
                NULL,  # inbound_links
            )
            # iterate through shortest paths and load the links with volume
            # link flow is thread local
            for j in range(number_of_targets):
                path = <igraph_vector_t*> vector_ptr_get(&paths_for_source, j)
                # TODO: (thread safely) add the path to the path set
                index = <long int> vector_get(trip_indices, j)
                volume = vector_get(demand.volumes.vec, index)
                # matrices are stored column-wise, grab a pointer to the first
                # element of the column for this thread
                _flow = <igraph_real_t*> igraph_matrix_e_ptr(&_flows, 0, thread_id)
                for k in range(vector_len(path)):
                    eid = <long int> vector_get(path, k)
                    _flow[eid] += volume
    # sum the thread local link flows into the final link flow
    for eid in range(vector_len(flow.vec)):
        _flow_at = 0.0
        for j in range(n_threads):
            _flow_at += igraph_matrix_e(&_flows, eid, j)
        vector_set(flow.vec, eid, _flow_at)
    igraph_matrix_destroy(&_flows)

cdef PointerVector init_path_vectors(OrgnDestDemand demand):
    cdef long int i, j, k, n_targets, n_sources = demand.number_of_sources()
    cdef PointerVector paths = PointerVector.nulls(demand.number_of_trips())
    igraph_vector_ptr_set_item_destructor(paths.vec,
                                          <igraph_finally_func_t*> igraph_vector_destroy)
    cdef igraph_vector_t* data = <igraph_vector_t*> malloc(demand.number_of_trips() * sizeof(igraph_vector_t))
    for i in range(n_sources):
        n_targets = demand.number_of_targets(i)
        for j in range(n_targets):
            k = demand.index_of(i, j)
            igraph_vector_init(data + k, 0)
            vector_ptr_set(paths.vec, k, data + k)
    return paths


def load_network(DiGraph network, Vector cost, OrgnDestDemand demand):
    cdef Vector flow = Vector.zeros(network.number_of_links())
    cdef long int i, number_of_sources = demand.number_of_sources()
    cdef PointerVector paths = init_path_vectors(demand)
    shortest_paths_assignment(network, cost, demand, flow, paths)
    return flow, paths