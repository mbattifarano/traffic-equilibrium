# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,

from .leveldb cimport *
from .igraph cimport igraph_vector_resize
from .igraph_utils cimport vector_len, vector_get, vector_set
from libc.stdlib cimport free, malloc
from libc.stdio cimport fprintf, printf, stderr, setbuf, stdout
from cython.parallel cimport prange
from openmp cimport omp_get_max_threads
from cpython cimport array

import array

setbuf(stdout, NULL)

cdef leveldb_options_t* default_options() nogil:
    cdef leveldb_options_t *options = leveldb_options_create()
    leveldb_options_set_create_if_missing(options, 1)
    #leveldb_options_set_write_buffer_size(options, 128 * 1024 * 1024)
    #leveldb_options_set_max_file_size(options, 128 * 1024 * 1024)
    leveldb_options_set_block_size(options, 8 * 1024 * 1024)
    return options


cdef class Cursor:
    def __cinit__(self):
        self.iterator = NULL
        self.counter = 0
        self.key_ptr = NULL
        self.key_len = 0
        self.val_ptr = NULL
        self.val_len = 0

    def __dealloc__(self):
        if self.iterator:
            leveldb_iter_destroy(self.iterator)

    cdef void attach(self, leveldb_t* db, leveldb_readoptions_t *readoptions):
        self.iterator = leveldb_create_iterator(db, readoptions)

    cpdef void populate(self):
        self.key_ptr = leveldb_iter_key(self.iterator,
                                        &self.key_len)
        self.val_ptr = leveldb_iter_value(self.iterator,
                                          &self.val_len)

    cpdef void reset(self):
        leveldb_iter_seek_to_first(self.iterator)
        self.counter = 0

    cpdef void next(self):
        leveldb_iter_next(self.iterator)
        self.counter += 1

    cpdef bint is_valid(self):
        return <bint> leveldb_iter_valid(self.iterator)

    cpdef bytes key(self):
        return <bytes> self.key_ptr[:self.key_len]

    cpdef bytes value(self):
        return <bytes> self.val_ptr[:self.val_len]


cdef class PathDB:
    def __cinit__(self, str fname):
        cdef char *error = NULL;
        self.name = fname
        self.options = default_options()
        self.db = leveldb_open(self.options, fname.encode('ascii'), &error)
        if error != NULL:
            fprintf(stderr, "ERROR (cinit): %s\n", error)
            free(error)
        self.readoptions = leveldb_readoptions_create()
        self.writeoptions = leveldb_writeoptions_create()
        leveldb_writeoptions_set_sync(self.writeoptions, 0)
        self.n_writers = omp_get_max_threads()
        self.writers = < leveldb_writebatch_t** > malloc(
            self.n_writers * sizeof(leveldb_writebatch_t*)
        )
        cdef int i
        for i in range(self.n_writers):
            self.writers[i] = leveldb_writebatch_create()

    def __dealloc__(self):
        cdef int i
        for i in range(self.n_writers):
            leveldb_writebatch_destroy(self.writers[i])
        leveldb_options_destroy(self.options)
        leveldb_readoptions_destroy(self.readoptions)
        leveldb_writeoptions_destroy(self.writeoptions)

    cpdef void destroy_db(self):
        self.close()
        cdef char *error = NULL
        leveldb_destroy_db(self.options, self.name.encode('ascii'), &error)
        if error != NULL:
            fprintf(stderr, "ERROR (destroy_db): %s\n", error)
            free(error)

    cpdef void close(self):
        leveldb_close(self.db)

    cdef void put(self, const bytes_t[:] key, const bytes_t[:] value):
        cdef char *error = NULL
        leveldb_put(self.db,
                    self.writeoptions,
                    <char *> &key[0], key.shape[0],
                    <char *> &value[0], value.shape[0],
                    &error)
        if error != NULL:
            fprintf(stderr, "ERROR (put): %s\n", error)
            free(error)

    cdef void commit(self) nogil:
        cdef int i
        cdef char *error = NULL
        cdef leveldb_writebatch_t *writer
        for i in range(self.n_writers):
            writer = self.writers[i]
            leveldb_write(self.db, self.writeoptions,
                          writer, &error)
            if error:
                fprintf(stderr, "ERROR (commit): %s\n", error)
                free(error)
                error = NULL
            leveldb_writebatch_clear(writer)

    cdef Value get(self, const bytes_t[:] key):
        cdef size_t length = 0
        cdef char *error = NULL
        cdef char *_value = NULL
        _value = leveldb_get(self.db, self.readoptions,
                             <char *> &key[0], key.shape[0],
                             &length, &error)
        if error != NULL:
            fprintf(stderr, "ERROR (get): %s\n", error)
            free(error)
        return Value.of(_value, length)

    def get_py(self, bytes key):
        cdef const bytes_t[:] _key = key
        return self.get(_key)

    def set_py(self, bytes key, bytes value):
        return self.put(key, value)

    cpdef Cursor cursor(self):
        cdef Cursor cursor = Cursor.__new__(Cursor)
        cursor.attach(self.db, self.readoptions)
        return cursor

    def items(self):
        cdef Cursor cursor = self.cursor()
        cursor.reset()
        while cursor.is_valid():
            cursor.populate()
            yield array.array('I', cursor.key()), array.array('l', cursor.value())
            cursor.next()
        del cursor


cdef class Value:
    @staticmethod
    cdef of(char* val, size_t length):
        cdef Value value = Value.__new__(Value)
        if val == NULL:
            value.present = False
        else:
            value.present = True
            value.value_ptr = val
            value.length = length
        return value

    def __dealloc__(self):
        free(self.value_ptr)

    def tobytes(self):
        cdef bytes value = self.value_ptr[:self.length]
        return value

cdef int write_path_to_batch(leveldb_writebatch_t *writer,
                             igraph_vector_t *path,
                             trip_id_storage_t trip_id) nogil:
    cdef long int i, n_links = vector_len(path)
    cdef size_t key_len = n_links * sizeof(link_id_storage_t)
    cdef link_id_storage_t *link_ids = <link_id_storage_t*> malloc(key_len)
    cdef size_t val_len = sizeof(trip_id_storage_t)
    for i in range(n_links):
        link_ids[i] = <link_id_storage_t> vector_get(path, i)
    leveldb_writebatch_put(
        writer,
        <char*> link_ids, key_len,
        <char*> &trip_id, val_len
    )
    free(link_ids)


cdef trip_id_storage_t read_path_from_entry(
        const char *key_ptr, size_t key_len,
        const char *val_ptr, size_t val_len,
        igraph_vector_t *path
) nogil:
    cdef size_t i, n_links = key_len / sizeof(link_id_storage_t)
    if val_len < sizeof(trip_id_storage_t):
        printf("ERROR: corrupt data; size of stored trip id is different from its storage type sizeof(trip_id_storage_t) = %lu; val_len = %lu.\n",
               sizeof(trip_id_storage_t), val_len)
    cdef trip_id_storage_t *trip_id = <trip_id_storage_t*> val_ptr
    cdef link_id_storage_t *link_ids = <link_id_storage_t*> key_ptr
    igraph_vector_resize(path, n_links)
    for i in range(n_links):
        vector_set(path, i, link_ids[i])
    return trip_id[0]

cdef trip_id_storage_t read_path_from_cursor(
        Cursor cursor,
        igraph_vector_t *path
):
    return read_path_from_entry(
        cursor.key_ptr, cursor.key_len,
        cursor.val_ptr, cursor.val_len,
        path
    )