# cython: boundscheck=False, wraparound=False, cdivision=True,
from .graph cimport graph_index_t
from .igraph cimport igraph_real_t, igraph_vector_t
from .igraph_utils cimport vector_set, vector_ptr_set, vector_len, vector_ptr_get, vector_get
from .vector cimport Vector, PointerVector


cdef class Trip:
    def __cinit__(self, graph_index_t source, graph_index_t target,
                  igraph_real_t volume):
        self.source = source
        self.target = target
        self.volume = volume


cdef class Trips:
    def __cinit__(self):
        self.trips = []
        self.source_index = {}

    cpdef void append(self, graph_index_t source, graph_index_t target, igraph_real_t volume):
        self.trips.append(Trip.__new__(Trip, source, target, volume))
        if source not in self.source_index:
            self.source_index[source] = {}
        self.source_index[source][target] = volume

    cpdef OrgnDestDemand compile(self):
        cdef long int number_of_sources = len(self.source_index)
        cdef long int number_of_trips = sum([len(ts) for ts in self.source_index.values()])
        cdef OrgnDestDemand od_demand = OrgnDestDemand.__new__(OrgnDestDemand, number_of_sources, number_of_trips)
        cdef long int i, number_of_targets, j, k = 0
        cdef Vector targets, trip_index
        for i, (source, target_volumes) in enumerate(self.source_index.items()):
            # store the source node
            vector_set(od_demand.sources.vec, i, source)
            number_of_targets = len(target_volumes)
            # Temporary vectors for targets and trip indices
            targets = Vector.zeros(number_of_targets, owner=False)
            trip_index = Vector.zeros(number_of_targets, owner=False)
            for j, (target, volume) in enumerate(target_volumes.items()):
                # store the target node in target vec
                vector_set(targets.vec, j, target)
                # store the volume in the volumes vec
                vector_set(od_demand.volumes.vec, k, volume)
                # store the trip index in the trip index vec
                vector_set(trip_index.vec, j, k)
                # increment the linear index
                k += 1
            vector_ptr_set(od_demand.targets.vec, i, targets.vec)
            vector_ptr_set(od_demand.trip_index.vec, i, trip_index.vec)
        return od_demand


cdef class OrgnDestDemand:
    def __cinit__(self, long int number_of_sources,
                  long int number_of_trips):
        self.sources = Vector.zeros(number_of_sources)
        self.targets = PointerVector.nulls(number_of_sources)
        self.volumes = Vector.zeros(number_of_trips)
        self.trip_index = PointerVector.nulls(number_of_sources)

    cdef long int number_of_sources(OrgnDestDemand self) nogil:
        return vector_len(self.sources.vec)

    cdef long int number_of_targets(OrgnDestDemand self, long int source) nogil:
        cdef igraph_vector_t* targets = <igraph_vector_t*> vector_ptr_get(self.targets.vec, source)
        return vector_len(targets)

    cdef long int number_of_trips(OrgnDestDemand self) nogil:
        return vector_len(self.volumes.vec)

    cdef long int index_of(OrgnDestDemand self, long int source, long int target) nogil:
        cdef igraph_vector_t* _index_by_target = <igraph_vector_t*> vector_ptr_get(self.trip_index.vec, source)
        return <long int> vector_get(_index_by_target, target)