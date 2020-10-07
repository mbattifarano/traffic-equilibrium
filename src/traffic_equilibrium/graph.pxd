# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from .igraph cimport igraph_t, igraph_integer_t, igraph_vector_t

ctypedef unsigned int graph_index_t

cdef class DiGraph:
    cdef igraph_t* graph
    cdef readonly str name

    cdef str get_name(DiGraph self)
    cdef void set_name(DiGraph self, str name)
    cdef graph_index_t number_of_nodes(DiGraph self) nogil
    cdef graph_index_t number_of_links(DiGraph self) nogil
    cdef void add_nodes(DiGraph self, igraph_integer_t number_of_nodes) nogil
    cdef void add_links(DiGraph self, igraph_vector_t* links) nogil
