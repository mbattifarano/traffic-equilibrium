from .igraph cimport igraph_vector_t, igraph_real_t
from .vector cimport Vector

cdef class LinkCost:
    cdef void compute_link_cost(LinkCost self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(LinkCost self,
                                          Vector flow)

cdef class LinkCostBPR(LinkCost):
    cdef igraph_real_t alpha
    cdef igraph_real_t beta
    cdef readonly Vector capacity
    cdef readonly Vector free_flow_travel_time

    cdef void compute_link_cost(LinkCostBPR self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)


    cpdef Vector compute_link_cost_vector(LinkCostBPR self,
                                          Vector flow)

cdef class LinkCostLinear(LinkCost):
    cdef readonly Vector coefficients
    cdef readonly Vector constants

    cdef void compute_link_cost(LinkCostLinear self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(LinkCostLinear self,
                                          Vector flow)
