from .vector cimport Vector
from .graph cimport DiGraph
from .trips cimport OrgnDestDemand
from .link_cost cimport LinkCost
from .pathdb cimport PathDB

cdef class Problem:
    cdef readonly DiGraph network
    cdef readonly OrgnDestDemand demand
    cdef public LinkCost cost_fn

cdef class FrankWolfeSettings:
    cdef readonly long int max_iterations
    cdef readonly double gap_tolerance
    cdef readonly double line_search_tolerance
    cdef readonly size_t commit_interval

cdef class Result:
    cdef readonly Problem problem
    cdef readonly FrankWolfeSettings settings
    cdef readonly PathDB paths
    cdef readonly Vector prev_flow
    cdef readonly Vector flow
    cdef readonly Vector cost
    cdef readonly Vector trip_cost
    cdef readonly double gap
    cdef readonly long int iterations
    cdef readonly double duration

cdef class ConvergenceCriteria:
    cdef readonly double relative_gap
    cdef readonly double average_excess_cost