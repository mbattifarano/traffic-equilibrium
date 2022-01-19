from .leveldb cimport *
from .igraph cimport igraph_vector_t

ctypedef unsigned char bytes_t
ctypedef unsigned int link_id_storage_t
ctypedef unsigned int trip_id_storage_t

cdef class Value:
    cdef char* value_ptr
    cdef size_t length
    cdef readonly bint present

    @staticmethod
    cdef of(char* val, size_t length)


cdef class Cursor:
    cdef leveldb_iterator_t* iterator
    cdef readonly unsigned long counter
    cdef const char* key_ptr
    cdef size_t key_len
    cdef const char* val_ptr
    cdef size_t val_len

    cdef void attach(self, leveldb_t *db, leveldb_readoptions_t *readoptions)
    cpdef void populate(self)
    cpdef void reset(self)
    cpdef void next(self)
    cpdef bint is_valid(self)
    cpdef bytes key(self)
    cpdef bytes value(self)

cdef class PathDB:
    cdef str name
    cdef leveldb_t* db
    cdef leveldb_writebatch_t** writers
    cdef int n_writers
    cdef leveldb_options_t* options
    cdef leveldb_readoptions_t* readoptions
    cdef leveldb_writeoptions_t* writeoptions

    cpdef void destroy_db(self)
    cpdef void close(self)
    cdef void commit(self) nogil
    cdef void put(self, const bytes_t[:] key, const bytes_t[:] value)
    cdef Value get(self, const bytes_t[:] key)
    cpdef Cursor cursor(self)

cdef int write_path_to_batch(leveldb_writebatch_t *writer,
                             igraph_vector_t *path,
                             trip_id_storage_t trip_id
                             ) nogil

cdef trip_id_storage_t read_path_from_entry(
        const char *key_ptr, size_t key_len,
        const char *val_ptr, size_t val_len,
        igraph_vector_t *path) nogil

cdef trip_id_storage_t read_path_from_cursor(
        Cursor cursor,
        igraph_vector_t *path
)
