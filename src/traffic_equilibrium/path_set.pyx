from libc.stdlib cimport free, malloc, calloc
from libc.stdio cimport stdout, setbuf, printf, fopen, fclose, FILE
from .dang cimport dang_t, dang_init, dang_print, dang_write, dang_write_gz, DANG_SET_SOURCE, dang_size, dang_upsert_path, dang_destroy
from .zlib cimport gzopen, gzclose, gzFile

import os
import yaml

import numpy as np


cdef extern from "<stdio.h>" nogil:
    FILE *popen(const char *command, const char *_type)
    int pclose(FILE *stream)


cdef class PathSet:

    def __cinit__(self, size_t number_of_sources):
        self.number_of_sources = number_of_sources
        self.tries = <dang_t*> calloc(number_of_sources, sizeof(dang_t))
        cdef size_t i = 0
        cdef dang_t *dang;
        for i in range(number_of_sources):
            dang = <dang_t*> self.tries + i
            dang_init(dang)
            DANG_SET_SOURCE(dang, i)

    def __dealloc__(self):
        cdef size_t i = 0
        if self.tries:
            for i in range(self.number_of_sources):
                dang_destroy(self.tries + i)
            free(self.tries)

    cdef dang_t* get_trie(self, size_t i) nogil:
        return self.tries + i

    def display(self):
        cdef size_t i = 0
        for i in range(self.number_of_sources):
            dang_print(self.get_trie(i))

    def write(self, dirname):
        fname = os.path.join(dirname, "paths.bin")
        cdef size_t i = 0
        cdef FILE *pipe = popen(f"gzip - > {fname}.gz".encode('ascii'), "w");
        for i in range(self.number_of_sources):
            dang_write(self.get_trie(i), pipe, False)
        pclose(pipe)

    def clear(self):
        cdef size_t i = 0
        cdef dang_t *dang
        if self.tries:
            for i in range(self.number_of_sources):
                dang = <dang_t*> self.tries + i
                dang_destroy(dang)
                DANG_SET_SOURCE(dang, i)

    @staticmethod
    def load(dirname):
        with open(os.path.join(dirname, "paths.yaml")) as fp:
            obj = yaml.safe_load(fp)
        cdef size_t number_of_sources = len(obj)
        cdef unsigned int source, target
        cdef dang_t *dang
        cdef PathSet path_set = PathSet.__new__(PathSet, number_of_sources)
        cdef unsigned int [:] edges
        # data: [source, [[target, [edge, ...]], ...]]
        for data in obj:
            source, paths = data
            dang = path_set.get_trie(source)
            for path in paths:
                target, _edges = path
                _edges = np.array(_edges, dtype=np.uintc)
                edges = _edges
                dang_upsert_path(dang, &edges[0], len(_edges), target)
        return path_set

    cdef long int memory_usage(self) nogil:
        cdef size_t i = 0
        cdef long int n_bytes = 0;
        for i in range(self.number_of_sources):
            n_bytes += dang_size(self.tries + i)
        return n_bytes

    cdef long int number_of_paths(self) nogil:
        cdef size_t i = 0
        cdef long int n_paths = 0
        for i in range(self.number_of_sources):
            n_paths += (self.tries + i).count
        return n_paths
