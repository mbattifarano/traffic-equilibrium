from libc.stdio cimport fprintf, stderr
from .gurobi cimport *


cdef class LPModel:

    def __cinit__(self):
        GRBloadenv(&self.env, "gurobi.log")
        GRBnewmodel(self.env, &self.model,
                    "model", 0,
                    NULL, NULL, NULL, NULL, NULL)

    cdef int add_variables(self, int n_variables,
                           size_t nnz, size_t *vbeg, int *vind, double *vval,
                           double *obj, double *lb, double *ub ):
        cdef int errno = GRBXaddvars(self.model, n_variables,
                           nnz, vbeg, vind, vval,
                           obj, lb, ub,
                           NULL,  # vtype
                           NULL   # varnames
                           )
        if errno:
            fprintf(stderr, "Error encountered during `GRBXaddvars`: (%i) %s\n",
                    errno, GRBgeterrormsg(self.env))
        return errno

    cdef int add_variable(self, int nnz, int *vind, double *vval,
                          double obj, double lb, double ub,
                          char vtype, const char *varname):
        cdef int errno = GRBaddvar(self.model,
                                   nnz, vind, vval,
                                   obj, lb, ub,
                                   vtype, varname)
        if errno:
            fprintf(stderr, "Error encountered during `GRBaddvar`: (%i) %s\n",
                    errno, GRBgeterrormsg(self.env))
        return errno


    cdef int add_empty_constraint(self, char sense, double rhs):
        cdef int errno
        errno = GRBaddconstr(self.model, 0, NULL, NULL, sense, rhs, NULL)
        if errno:
            fprintf(stderr, "Error encountered during `GRBaddconstr`: (%i) %s\n",
                    errno, GRBgeterrormsg(self.env))
        return errno

    cpdef int update(self):
        return GRBupdatemodel(self.model)

    cpdef int optimize(self):
        return GRBoptimize(self.model)

    cpdef int n_constraints(self):
        cdef int n
        GRBgetintattr(self.model, "NumConstrs", &n)
        return n

    cpdef int n_variables(self):
        cdef int n
        GRBgetintattr(self.model, "NumVars", &n)
        return n

    cpdef int write(self, str fname):
        return GRBwrite(self.model, fname.encode('ascii'))

    cpdef void get_values(self, double [:] result):
        cdef size_t i, n = self.n_variables()
        for i in range(n):
            GRBgetdblattrelement(self.model, "X", i, &result[i])
