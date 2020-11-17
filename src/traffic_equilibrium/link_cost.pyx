# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from .igraph cimport *
from .igraph_utils cimport igraph_vector_pow

import os
import json
import numpy as np

cdef class LinkCostFunctions:
    @classmethod
    def get(cls, name, kwargs):
        pass


cdef class LinkCost:

    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        pass

    cpdef Vector compute_link_cost_vector(self, Vector flow):
        cdef Vector cost = Vector.zeros(len(flow))
        self.compute_link_cost(flow.vec, cost.vec)
        return cost

    def save(self, name):
        return NotImplemented

    @classmethod
    def load(cls, name):
        with open(os.path.join(name, "cost_function.json")) as fp:
            obj = json.load(fp)
        class_name = obj['name']
        cls_ = {
            'bpr': LinkCostBPR,
            'marginal_bpr': LinkCostMarginalBPR,
            'linear': LinkCostLinear,
            'marginal_linear': LinkCostMarginalLinear
        }.get(class_name.lower())
        if cls_ is None:
            raise Exception(f"Unrecognized class name {class_name}.")
        return cls_.load(obj['kwargs'])

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
        self.name = "bpr"

    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        # all operations write into the first arg
        # accumulate everything into cost
        igraph_vector_update(cost, flow)  # cost = flow
        igraph_vector_div(cost, self.capacity.vec)   # cost = flow/capacity
        igraph_vector_pow(cost, self.beta)       # cost = (flow/capacity)**beta
        igraph_vector_scale(cost, self.alpha)    # cost = alpha*(flow/capacity)**beta
        igraph_vector_add_constant(cost, 1.0) # cost = 1 + alpha*(flow/capacity)**beta
        igraph_vector_mul(cost, self.free_flow_travel_time.vec)   # cost = freeflow * (1 + alpha*(flow/capacity)**beta)

    def save(self, name):
        with open(os.path.join(name, "cost_function.json"), "w") as fp:
            json.dump({
                'name': self.name,
                'kwargs': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'capacity': self.capacity.to_array().tolist(),
                    'free_flow_travel_time': self.free_flow_travel_time.to_array().tolist(),
                }
            }, fp, indent=2)

    @classmethod
    def load(cls, kwargs):
        cdef Vector cap = Vector.copy_of(np.array(kwargs['capacity']))
        cdef Vector fftt = Vector.copy_of(np.array(kwargs['free_flow_travel_time']))
        return cls(kwargs['alpha'], kwargs['beta'], cap, fftt)


cdef class LinkCostMarginalBPR(LinkCostBPR):
    def __cinit__(self,
                  igraph_real_t alpha,
                  igraph_real_t beta,
                  Vector capacity,
                  Vector free_flow_travel_time):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.free_flow_travel_time = free_flow_travel_time
        self.name = "marginal_bpr"

    cdef void compute_link_cost(self, igraph_vector_t *flow,
                                igraph_vector_t *cost):
        """Compute the marginal link cost of the BPR link cost function."""
        igraph_vector_update(cost, flow)  # cost = flow
        igraph_vector_div(cost, self.capacity.vec)   # cost = flow/capacity
        igraph_vector_pow(cost, self.beta)       # cost = (flow/capacity)**beta
        igraph_vector_scale(cost, self.alpha)    # cost = alpha*(flow/capacity)**beta
        igraph_vector_scale(cost, self.beta + 1.0)  # cost = (1 + beta)*alpha*(flow/capacity)**beta
        igraph_vector_add_constant(cost, 1.0)  # cost = 1 + (1 + beta)*alpha*(flow/capacity)**beta
        igraph_vector_mul(cost, self.free_flow_travel_time.vec)   # cost = freeflow * (1 + (1 + beta)*alpha*(flow/capacity)**beta)

cdef class LinkCostLinear(LinkCost):
    def __cinit__(self, Vector coefficients, Vector constants):
        self.coefficients = coefficients
        self.constants = constants
        self.name = "linear"

    cdef void compute_link_cost(self, igraph_vector_t* flow, igraph_vector_t* cost):
        igraph_vector_update(cost, flow)
        igraph_vector_mul(cost, self.coefficients.vec)
        igraph_vector_add(cost, self.constants.vec)

    def save(self, name):
        with open(os.path.join(name, "cost_function.json"), "w") as fp:
            json.dump({
                'name': self.name,
                'kwargs': {
                    'coefficients': self.coefficients.to_array().tolist(),
                    'constants': self.constants.to_array().tolist(),
                }
            }, fp, indent=2)

    @classmethod
    def load(cls, kwargs):
        cdef Vector coefs = Vector.copy_of(np.array(kwargs['coefficients']))
        cdef Vector constants = Vector.copy_of(np.array(kwargs['constants']))
        return cls(coefs, constants)


cdef class LinkCostMarginalLinear(LinkCostLinear):
    def __cinit__(self, Vector coefficients, Vector constants):
        self.coefficients = coefficients
        self.constants = constants
        self.name = "marginal_linear"

    cdef void compute_link_cost(self, igraph_vector_t *flow,
                                         igraph_vector_t *cost):
        igraph_vector_update(cost, flow)  # cost = flow
        igraph_vector_mul(cost, self.coefficients.vec)  # cost = flow * coeffs
        igraph_vector_scale(cost, 2.0)  # cost = 2 * flow * coeffs
        igraph_vector_add(cost, self.constants.vec)  # cost = 2 * flow * coeffs + constants
