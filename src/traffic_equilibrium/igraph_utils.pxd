# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.math cimport pow
from cython.parallel cimport prange
from .igraph cimport (
    igraph_vector_t,
    igraph_real_t,
    igraph_vector_long_t,
    igraph_vector_bool_t,
    igraph_vector_ptr_t,
)

# vectors
cdef inline igraph_real_t vector_get(igraph_vector_t* vector, long int at) nogil:
    return vector.stor_begin[at]

cdef inline void vector_set(igraph_vector_t* vector, long int at, igraph_real_t value) nogil:
    vector.stor_begin[at] = value

cdef inline void vector_inc(igraph_vector_t* vector, long int at, igraph_real_t value) nogil:
    vector.stor_begin[at] += value

cdef inline long int vector_len(igraph_vector_t* vector) nogil:
    return vector.end - vector.stor_begin

cdef inline void igraph_vector_pow(igraph_vector_t* vector, igraph_real_t exponent) nogil:
    cdef long int i, length = vector_len(vector)
    cdef igraph_real_t val
    for i in range(length):
        val = pow(vector_get(vector, i), exponent)
        vector_set(vector, i, val)

cdef inline igraph_real_t igraph_vector_dot(igraph_vector_t* v1, igraph_vector_t* v2) nogil:
    cdef long int i, n = vector_len(v1)
    cdef igraph_real_t value = 0.0
    for i in prange(n):
        value += vector_get(v1, i) * vector_get(v2, i)
    return value

# long vectors
cdef inline long int vector_long_get(igraph_vector_long_t* vector, long int at) nogil:
    return vector.stor_begin[at]

cdef inline void vector_long_set(igraph_vector_long_t* vector, long int at, long int value) nogil:
    vector.stor_begin[at] = value

cdef inline void vector_long_inc(igraph_vector_long_t* vector, long int at, long int value) nogil:
    vector.stor_begin[at] += value

cdef inline long int vector_long_len(igraph_vector_long_t* vector) nogil:
    return vector.end - vector.stor_begin

# bool vectors
cdef inline bint vector_bool_get(igraph_vector_bool_t* vector, long int at) nogil:
    return vector.stor_begin[at]

cdef inline void vector_bool_set(igraph_vector_bool_t* vector, long int at, bint value) nogil:
    vector.stor_begin[at] = value

cdef inline void vector_bool_set_true(igraph_vector_bool_t* vector, long int at) nogil:
    vector.stor_begin[at] = 1

cdef inline long int vector_bool_len(igraph_vector_bool_t* vector) nogil:
    return vector.end - vector.stor_begin

# ptr vectors
cdef inline void* vector_ptr_get(igraph_vector_ptr_t* vector, long int at) nogil:
    return vector.stor_begin[at]

cdef inline void vector_ptr_set(igraph_vector_ptr_t* vector, long int at, void* value) nogil:
    vector.stor_begin[at] = value

cdef inline void** vector_ptr_get_addr(igraph_vector_ptr_t* vector, long int at) nogil:
    return vector.stor_begin + at

cdef inline long int vector_ptr_len(igraph_vector_ptr_t* vector) nogil:
    return vector.end - vector.stor_begin