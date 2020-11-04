from libc.stdlib cimport free, malloc, calloc
from libc.stdio cimport stdout, setbuf, printf, fopen, fclose, FILE
from .dang cimport dang_t, dang_init, dang_print, dang_write, DANG_SET_SOURCE, dang_size

import os

cdef class PathSet:

    def __cinit__(self, long int number_of_sources):
        self.number_of_sources = number_of_sources
        self.tries = <dang_t*> calloc(self.number_of_sources, sizeof(dang_t))
        cdef long int i = 0
        cdef dang_t *dang;
        for i in range(self.number_of_sources):
            dang = <dang_t*> self.tries + i
            dang_init(dang)
            DANG_SET_SOURCE(dang, i)

    def __dealloc__(self):
        if self.tries:
            free(self.tries)

    cdef dang_t* get_trie(self, long int i) nogil:
        return self.tries + i

    def display(self):
        cdef long int i = 0
        for i in range(self.number_of_sources):
            dang_print(self.get_trie(i))

    def write(self, dirname):
        cdef long int i = 0
        cdef FILE *fp = fopen(os.path.join(dirname, "paths.yaml").encode("ascii"), b"w")
        for i in range(self.number_of_sources):
            dang_write(self.get_trie(i), fp, False)
        fclose(fp)

    cdef long int memory_usage(self) nogil:
        cdef long int i = 0, n_bytes = 0;
        for i in range(self.number_of_sources):
            n_bytes += dang_size(self.tries + i)
        return n_bytes

    cdef long int number_of_paths(self) nogil:
        cdef long int i = 0, n_paths = 0;
        for i in range(self.number_of_sources):
            n_paths += (self.tries + i).count
        return n_paths
