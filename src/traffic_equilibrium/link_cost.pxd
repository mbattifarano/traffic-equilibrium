from .igraph cimport igraph_vector_t, igraph_real_t
from .vector cimport Vector

# TODO gradient
cdef class LinkCost:
    cdef readonly str name
    cdef void compute_link_cost(LinkCost self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cdef void gradient(self,
                       igraph_vector_t* flow,
                       igraph_vector_t* grad)

    cpdef Vector compute_link_cost_vector(LinkCost self,
                                          Vector flow)

cdef class LinkCostBPR(LinkCost):
    cdef readonly igraph_real_t alpha
    cdef readonly igraph_real_t beta
    cdef readonly Vector capacity
    cdef readonly Vector free_flow_travel_time

    cdef void compute_link_cost(LinkCostBPR self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(LinkCostBPR self, Vector flow)

cdef class LinkCostMarginalBPR(LinkCostBPR):
    cdef void compute_link_cost(self, igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(self, Vector flow)

cdef class LinkCostLinear(LinkCost):
    cdef readonly Vector coefficients
    cdef readonly Vector constants

    cdef void compute_link_cost(LinkCostLinear self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(LinkCostLinear self, Vector flow)

cdef class LinkCostMarginalLinear(LinkCostLinear):
    cdef void compute_link_cost(self,
                                igraph_vector_t* flow,
                                igraph_vector_t* cost)

    cpdef Vector compute_link_cost_vector(self, Vector flow)


