from .gurobi cimport *

cdef class LPModel:
    cdef GRBenv *env
    cdef GRBmodel *model

    cdef int add_variables(self, int n_variables,
                           size_t nnz,
                           size_t *vbeg,
                           int *vind,
                           double *vval,
                           double *obj,
                           double *lb,
                           double *ub)

    cdef int add_variable(self, int nnz, int *vind, double *vval,
                          double obj, double lb, double ub,
                          char vtype, const char *varname)

    cdef int add_empty_constraint(self, char sense, double rhs)

    cpdef int optimize(self)
    cpdef void get_values(self, double[:] result)
    cpdef int write(self, str fname)
    cpdef int n_constraints(self)
    cpdef int n_variables(self)
    cpdef int update(self)
