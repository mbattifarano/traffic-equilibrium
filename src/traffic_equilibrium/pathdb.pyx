from .leveldb cimport *
from libc.stdlib cimport free
from libc.stdio cimport fprintf, printf, stderr, setbuf, stdout

setbuf(stdout, NULL)

cdef leveldb_options_t* default_options() nogil:
    cdef leveldb_options_t *options = leveldb_options_create()
    cdef unsigned char t = 1
    leveldb_options_set_create_if_missing(options, t)
    leveldb_options_set_write_buffer_size(options, 4 * 1024 * 1024)
    leveldb_options_set_max_file_size(options, 2 * 1024 * 1024)
    return options


cdef class Cursor:
    def __cinit__(self):
        self.iterator = NULL
        self.counter = 0
        self.key_ptr = NULL
        self.key_len = 0
        self.val_ptr = NULL
        self.val_len = 0

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
        self.writebatch = leveldb_writebatch_create()

    def __dealloc__(self):
        leveldb_writebatch_clear(self.writebatch)
        leveldb_writebatch_destroy(self.writebatch)
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
        self.commit()
        leveldb_close(self.db)

    cdef void put(self, const bytes_t[:] key, const bytes_t[:] value):
        leveldb_writebatch_put(self.writebatch,
                               <char *> &key[0], key.shape[0],
                               <char *> &value[0], value.shape[0])

    cdef void commit(self):
        cdef char *error = NULL
        leveldb_write(self.db, self.writeoptions, self.writebatch, &error)
        if error != NULL:
            fprintf(stderr, "ERROR (commit): %s\n", error)
            free(error)
        leveldb_writebatch_clear(self.writebatch)

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

    def flush(self):
        self.commit()

    cpdef Cursor cursor(self):
        cdef Cursor cursor = Cursor.__new__(Cursor)
        cursor.attach(self.db, self.readoptions)
        return cursor


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
