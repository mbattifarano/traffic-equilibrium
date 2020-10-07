from .igraph cimport igraph_real_t
from .graph cimport graph_index_t
from .vector cimport Vector, PointerVector


cdef class Trip:
    cdef readonly graph_index_t source
    cdef readonly graph_index_t target
    cdef readonly igraph_real_t volume


cdef class Trips:
    cdef readonly list trips
    cdef dict source_index

    cpdef void append(self, graph_index_t source, graph_index_t target, igraph_real_t volume)
    cpdef OrgnDestDemand compile(self)


cdef class OrgnDestDemand:
    cdef readonly Vector sources
    cdef readonly PointerVector targets
    cdef readonly Vector volumes
    cdef readonly PointerVector trip_index

    cdef long int number_of_sources(OrgnDestDemand self) nogil
    cdef long int number_of_targets(OrgnDestDemand self, long int source) nogil
    cdef long int number_of_trips(OrgnDestDemand self) nogil
    cdef long int index_of(OrgnDestDemand self, long int source, long int target) nogil
