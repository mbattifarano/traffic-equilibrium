# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from .igraph cimport *
from .igraph_utils cimport igraph_vector_pow


cdef class LinkCost:
    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        return

    cpdef Vector compute_link_cost_vector(self, Vector flow):
        cdef Vector cost = Vector.zeros(len(flow))
        self.compute_link_cost(flow.vec, cost.vec)
        return cost

    def save(self, name):
        pass

    @classmethod
    def load(cls, name):
        pass

cdef class LinkCostBPR(LinkCost):
    def __cinit__(self,
                  igraph_real_t alpha,
                  igraph_real_t beta,
                  Vector capacity,
                  Vector free_flow_travel_time):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.free_flow_travel_time = free_flow_travel_time

    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        # all operations write into the first arg
        # accumulate everything into cost
        igraph_vector_update(cost, flow)  # cost = flow
        igraph_vector_div(cost, self.capacity.vec)   # cost = flow/capacity
        igraph_vector_pow(cost, self.beta)       # cost = (flow/capacity)**beta
        igraph_vector_scale(cost, self.alpha)    # cost = alpha*(flow/capacity)**beta
        igraph_vector_add_constant(cost, 1.0) # cost = 1 + alpha*(flow/capacity)**beta
        igraph_vector_mul(cost, self.free_flow_travel_time.vec)   # cost = freeflow * (1 + alpha*(flow/capacity)**beta)


cdef class LinkCostLinear(LinkCost):
    def __cinit__(self, Vector coefficients, Vector constants):
        self.coefficients = coefficients
        self.constants = constants

    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        igraph_vector_update(cost, flow)
        igraph_vector_mul(cost, self.coefficients.vec)
        igraph_vector_add(cost, self.constants.vec)