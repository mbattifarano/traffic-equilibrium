from .igraph cimport igraph_vector_t, igraph_vector_ptr_t


cdef class Vector:
    cdef igraph_vector_t* vec
    cdef bint owner

    @staticmethod
    cdef Vector zeros(unsigned int length, bint owner=*)

    @staticmethod
    cdef Vector of(igraph_vector_t* vec)


cdef class PointerVector:
    cdef igraph_vector_ptr_t* vec
    cdef bint owner

    @staticmethod
    cdef PointerVector nulls(unsigned int length, bint owner=*)
