from libc.stdio cimport FILE
from .zlib cimport gzFile

cdef extern from "dang.h" nogil:
    ctypedef unsigned int dang_index_t

    cdef struct dang:
        dang_index_t count
        dang_index_t nodes

    cdef struct dang_node:
        pass

    ctypedef dang_node dang_node_t

    ctypedef dang dang_t

    cdef void DANG_SET_SOURCE(dang_t *dang, dang_index_t source)

    cdef int dang_init(dang_t *dang)

    cdef int dang_destroy(dang_t *dang)

    cdef dang_node_t** dang_append_to_path(dang_t *dang,
                                           dang_node_t** walk,
                                           dang_index_t edge_id,
                                           long int *i)

    cdef int dang_mark_end(dang_t *dang,
                           dang_node_t** walk,
                           dang_index_t trip_id)

    cdef int dang_upsert_path(dang_t* dang,
                              dang_index_t* edge_id,
                              long int length,
                              dang_index_t trip_id)

    cdef int dang_write(dang_t *dang, FILE *stream, bint destructive)

    cdef int dang_write_gz(dang_t *dang, gzFile stream, bint destructive)

    cdef int dang_print(dang_t *dang)

    cdef long int dang_size(dang_t *dang)