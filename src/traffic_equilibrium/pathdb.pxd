from .leveldb cimport *

ctypedef unsigned char bytes_t;

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
    cdef leveldb_options_t* options
    cdef leveldb_readoptions_t* readoptions
    cdef leveldb_writeoptions_t* writeoptions
    cdef leveldb_writebatch_t* writebatch

    cpdef void destroy_db(self)
    cpdef void close(self)
    cdef void put(self, const bytes_t[:] key, const bytes_t[:] value)
    cdef void commit(self)
    cdef Value get(self, const bytes_t[:] key)
    cpdef Cursor cursor(self)
