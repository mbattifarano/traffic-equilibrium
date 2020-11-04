from .dang cimport dang_t

ctypedef unsigned int trie_index_t

cdef class PathSet:
    cdef readonly long int number_of_sources
    cdef dang_t* tries

    cdef dang_t* get_trie(self, long int i) nogil
    cdef long int memory_usage(self) nogil
    cdef long int number_of_paths(self) nogil
