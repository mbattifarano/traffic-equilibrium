# cython: boundscheck=False, wraparound=False, cdivision=True,
from libc.stdlib cimport free, malloc
from libc.stdio cimport stdout, setbuf, printf
cimport numpy
from .igraph cimport (
    igraph_real_t,
    igraph_vector_t,
    igraph_vector_ptr_t,
    igraph_vector_init,
    igraph_vector_ptr_init,
    igraph_vector_init_copy,
    igraph_vector_view,
    igraph_vector_destroy,
igraph_vector_ptr_destroy_all,
    igraph_vector_size,
)
from .igraph_utils cimport vector_get, vector_len

cimport numpy as np
import numpy as np

# disable stdout buffer (for debugging)
# setbuf(stdout, NULL)

cdef class Vector:
    def __cinit__(self):
        self.owner = False

    def __dealloc__(self):
        if self.vec is not NULL and self.owner is True:
            #printf("Freeing a vector at %p...", self.vec)
            igraph_vector_destroy(self.vec)
            free(self.vec)
            #printf("vector freed\n")
            self.vec = NULL

    @staticmethod
    cdef Vector zeros(unsigned int length, bint owner = True):
        cdef Vector vector = Vector.__new__(Vector)
        cdef igraph_vector_t* vec = <igraph_vector_t*> malloc(sizeof(igraph_vector_t))
        igraph_vector_init(vec, length)
        vector.vec = vec
        vector.owner = owner
        return vector

    @staticmethod
    cdef Vector of(igraph_vector_t* vec):
        cdef Vector vector = Vector.__new__(Vector)
        vector.vec = vec
        vector.owner = False
        return vector

    @staticmethod
    def view_of(numpy.ndarray[igraph_real_t, ndim=1] array):
        cdef Vector vector = Vector.__new__(Vector)
        cdef igraph_vector_t* vec = <igraph_vector_t*> malloc(sizeof(igraph_vector_t))
        cdef igraph_real_t[:] view = array
        cdef long int length = len(array)
        igraph_vector_view(vec, &view[0], length)
        vector.vec = vec
        vector.owner = False
        return vector

    @staticmethod
    def copy_of(numpy.ndarray[igraph_real_t, ndim=1] array):
        cdef Vector vector = Vector.__new__(Vector)
        cdef igraph_vector_t* vec = <igraph_vector_t*> malloc(sizeof(igraph_vector_t))
        cdef igraph_real_t[:] view = array
        cdef long int length = len(array)
        igraph_vector_init_copy(vec, &view[0], length)
        vector.vec = vec
        vector.owner = True
        return vector

    def to_array(self):
        cdef long int i, n = igraph_vector_size(self.vec)
        a = np.empty(n, dtype=np.double)
        cdef double[:] _a = a
        for i in range(n):
            _a[i] = vector_get(self.vec, i)
        return a

    def as_array(self):
        return np.asarray(<np.double_t[:vector_len(self.vec)]> self.vec.stor_begin)

    def to_list(self):
        return self.to_array().tolist()

    def __len__(self):
        return igraph_vector_size(self.vec)

    def __iter__(self):
        it = []
        cdef long int i, n = len(self)
        for i in range(n):
            it.append(vector_get(self.vec, i))
        return iter(it)

    def debug(self):
        return f"{len(self)} elements: {list(self)}"


cdef class PointerVector:
    def __cinit__(self):
        self.owner = False

    def __dealloc__(self):
        if self.vec is not NULL and self.owner is True:
            #printf("Freeing a pointer vector @ %p...", self.vec)
            igraph_vector_ptr_destroy_all(self.vec)
            #printf("Attempting to free the pointer...")
            free(self.vec)
            #printf("Vector freed\n")
            self.vec = NULL

    @staticmethod
    cdef PointerVector nulls(unsigned int length, bint owner = True):
        cdef PointerVector vector = PointerVector.__new__(PointerVector)
        cdef igraph_vector_ptr_t* vec = <igraph_vector_ptr_t*> malloc(sizeof(igraph_vector_ptr_t))
        igraph_vector_ptr_init(vec, length)
        vector.vec = vec
        vector.owner = owner
        return vector
