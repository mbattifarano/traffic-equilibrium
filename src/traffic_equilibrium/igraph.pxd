from libc.stdio cimport FILE

cdef extern from "igraph.h" nogil:
    cdef struct igraph_s:
        pass

    ctypedef igraph_s igraph_t

    ctypedef int igraph_integer_t
    ctypedef double igraph_real_t
    ctypedef bint igraph_bool_t

    ctypedef struct igraph_matrix_t:
        pass

    ctypedef struct igraph_deque_t:
        pass

    ctypedef struct igraph_vector_t:
        igraph_real_t* stor_begin
        igraph_real_t* stor_end
        igraph_real_t* end

    ctypedef struct igraph_vector_long_t:
        long int* stor_begin
        long int* stor_end
        long int* end

    ctypedef struct igraph_vector_bool_t:
        igraph_bool_t* stor_begin
        igraph_bool_t* stor_end
        igraph_bool_t* end

    ctypedef struct igraph_vector_ptr_t:
        void** stor_begin
        void** stor_end
        void** end

    ctypedef struct igraph_vs_t:
        pass

    ctypedef struct igraph_stack_t:
        pass

    ctypedef enum igraph_neimode_t:
        IGRAPH_OUT = 1

    ctypedef struct igraph_spmatrix_t:
        pass

    cdef struct s_spmatrix_iter:
        igraph_spmatrix_t *m
        long int pos
        long int ri
        long int ci
        igraph_real_t value

    ctypedef s_spmatrix_iter igraph_spmatrix_iter_t

    ctypedef struct igraph_finally_func_t:
        pass

    # matrix stuff
    int igraph_matrix_init(igraph_matrix_t *m, long int nrow, long int ncol)
    igraph_real_t* igraph_matrix_e_ptr(const igraph_matrix_t *m, long int row, long int col)
    igraph_real_t igraph_matrix_e(const igraph_matrix_t *m, long int row, long int col)
    void igraph_matrix_destroy(igraph_matrix_t *m)

    # sparse matrix stuff
    int igraph_spmatrix_init(igraph_spmatrix_t* m, long int nrow, long int ncol)
    void igraph_spmatrix_destroy(igraph_spmatrix_t *m)
    int igraph_spmatrix_set(igraph_spmatrix_t *m, long int row, long int col, igraph_real_t value)

    int igraph_spmatrix_iter_create(igraph_spmatrix_iter_t *mit, const igraph_spmatrix_t *m)
    igraph_bool_t igraph_spmatrix_iter_end(igraph_spmatrix_iter_t *mit)
    int igraph_spmatrix_iter_next(igraph_spmatrix_iter_t *mit)

    # vector stuff
    int igraph_vector_init(igraph_vector_t* vector, long int length)
    int igraph_vector_init_seq(igraph_vector_t* vector, igraph_real_t start, igraph_real_t stop)
    int igraph_vector_copy(igraph_vector_t* target, igraph_vector_t* source)
    void igraph_vector_destroy(igraph_vector_t* vector)

    int igraph_vector_ptr_init(igraph_vector_ptr_t* vector, long int length)
    igraph_vector_ptr_t* igraph_vector_ptr_view (igraph_vector_ptr_t *v, void **data, long int length)
    void igraph_vector_ptr_null(igraph_vector_ptr_t* vector)
    void igraph_vector_ptr_set(igraph_vector_ptr_t* vector, long int at, void* value)
    void* igraph_vector_ptr_e(igraph_vector_ptr_t* vector, long int at)
    long int igraph_vector_ptr_size(igraph_vector_ptr_t* vector)
    int igraph_vector_ptr_append(igraph_vector_ptr_t* source, igraph_vector_ptr_t* target )
    int igraph_vector_ptr_push_back(igraph_vector_ptr_t* vector, void* element)
    void* igraph_vector_ptr_pop_back(igraph_vector_ptr_t* vector)
    int igraph_vector_ptr_reserve(igraph_vector_ptr_t* vector, long int size)
    int igraph_vector_ptr_resize(igraph_vector_ptr_t* vector, long int newsize)
    igraph_finally_func_t* igraph_vector_ptr_set_item_destructor(igraph_vector_ptr_t *v, igraph_finally_func_t *func)
    void igraph_vector_ptr_destroy_all(igraph_vector_ptr_t* v)

    int igraph_vector_bool_init(igraph_vector_bool_t* vector, long int length)
    int igraph_vector_bool_push_back(igraph_vector_bool_t* vector, igraph_bool_t element)
    igraph_bool_t igraph_vector_bool_pop_back(igraph_vector_bool_t* vector)

    int igraph_vector_long_init(igraph_vector_long_t* vector, long int length)
    int igraph_vector_long_copy(igraph_vector_long_t* target, igraph_vector_long_t* source)
    int igraph_vector_long_push_back(igraph_vector_long_t* vector, long int element)
    long int igraph_vector_long_pop_back(igraph_vector_long_t* vector)
    int igraph_vector_long_reserve(igraph_vector_long_t* vector, long int size)
    long int igraph_vector_long_e(igraph_vector_long_t* vector, long int at)
    void igraph_vector_long_set(igraph_vector_long_t* vector, long int at, long int value)
    long int igraph_vector_long_size(igraph_vector_long_t* vector)
    int igraph_vector_long_resize(igraph_vector_long_t* vector, long int newsize)
    long int igraph_vector_long_max(igraph_vector_long_t* vector)
    bint igraph_vector_long_empty(igraph_vector_long_t* vector)
    void igraph_vector_long_destroy(igraph_vector_long_t* vector)
    void igraph_vector_long_clear(igraph_vector_long_t* vector)
    void igraph_vector_long_null(igraph_vector_long_t* vector)

    void igraph_vector_fill(igraph_vector_t* vector, igraph_real_t value)
    long int igraph_vector_size(igraph_vector_t* vector)
    igraph_bool_t igraph_vector_empty(igraph_vector_t*)

    igraph_vector_t* igraph_vector_view(igraph_vector_t* vec, igraph_real_t* data, long int length)
    int igraph_vector_init_copy(igraph_vector_t *v, igraph_real_t *data, long int length);

    igraph_real_t igraph_vector_e(igraph_vector_t* vector, long int position)
    void igraph_vector_set(igraph_vector_t* vector, long int position, igraph_real_t value)
    int igraph_vector_update(igraph_vector_t* target, igraph_vector_t* source)

    void igraph_vector_add_constant(igraph_vector_t* v, igraph_real_t value)
    void igraph_vector_scale(igraph_vector_t* v, igraph_real_t value)
    int igraph_vector_add(igraph_vector_t* v1, igraph_vector_t* v2)
    int igraph_vector_sub(igraph_vector_t* v1, igraph_vector_t* v2)
    int igraph_vector_mul(igraph_vector_t* v1, igraph_vector_t* v2)
    int igraph_vector_div(igraph_vector_t* v1, igraph_vector_t* v2)

    int igraph_vector_sum(igraph_vector_t* v)
    igraph_real_t igraph_vector_min(igraph_vector_t* v)
    igraph_real_t igraph_vector_max(igraph_vector_t* v)

    # graph construction
    int igraph_empty(igraph_t* graph, igraph_integer_t n, igraph_bool_t directed)
    int igraph_ring(igraph_t* graph, igraph_integer_t n, igraph_bool_t directed, igraph_bool_t mutual, igraph_bool_t circular)
    int igraph_destroy(igraph_t* graph)

    # graph inspection
    igraph_integer_t igraph_vcount(igraph_t *graph)
    igraph_integer_t igraph_ecount(igraph_t *graph)
    int igraph_edge(igraph_t* graph, igraph_integer_t eid,
                    igraph_integer_t* u, igraph_integer_t* v)
    int igraph_degree(igraph_t* graph, igraph_vector_t* result, igraph_vs_t vids,
                      igraph_neimode_t mode, igraph_bool_t loops)

    int igraph_vs_size(igraph_t* graph, igraph_vs_t* vs, igraph_integer_t* result)
    int igraph_vs_vector(igraph_vs_t* vs, igraph_vector_t* vec)
    int igraph_vs_all(igraph_vs_t* vs)
    igraph_vs_t igraph_vss_1(igraph_integer_t vid)
    igraph_vs_t igraph_vss_vector(igraph_vector_t* vec)
    void igraph_vs_destroy(igraph_vs_t* vs)

    # graph storage
    int igraph_write_graph_edgelist(igraph_t* graph, FILE* outstream)
    int igraph_read_graph_edgelist(igraph_t* graph, FILE* instream,
                                   igraph_integer_t n, igraph_bool_t directed)

    # graph modification
    int igraph_add_vertices(igraph_t* graph, igraph_integer_t number_of_nodes, void* attr)
    int igraph_add_edges(igraph_t* graph, igraph_vector_t* node_pairs, void* attr)

    # shortest paths
    int igraph_get_shortest_paths_dijkstra(igraph_t* graph,
                                           igraph_vector_ptr_t* nodes,
                                           igraph_vector_ptr_t* links,
                                           igraph_integer_t source,
                                           igraph_vs_t targets,
                                           igraph_vector_t *weights,
                                           igraph_neimode_t mode,
                                           igraph_vector_long_t* predeccessors,
                                           igraph_vector_long_t* inbound_links
                                           )
