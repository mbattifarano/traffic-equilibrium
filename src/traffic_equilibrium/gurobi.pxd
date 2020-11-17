cdef extern from "gurobi_c.h" nogil:
    ctypedef struct GRBenv:
        pass

    ctypedef struct GRBmodel:
        pass

    # constants
    char GRB_EQUAL
    char GRB_CONTINUOUS
    char GRB_INFINITY


    int GRBloadenv(GRBenv **env,
                   const char *logfilename)

    void GRBfreeenv(GRBenv *env)

    char* GRBgeterrormsg(GRBenv *env)

    int GRBnewmodel(GRBenv *env,
                    GRBmodel **model,
                    const char *name,
                    int numvars,
                    double *obj,
                    double *lb,
                    double *ub,
                    char *vtype,
                    const char **varnames
                    )

    int GRBwrite(GRBmodel *model, const char* filename)

    int GRBoptimize(GRBmodel *model)

    int GRBXaddvars(GRBmodel *model,
                    int numvars,
                    size_t numnz,
                    size_t *vbeg,  # variable i: vbeg[i] start position of nonzeros in vind, vval
                    int *vind,     # vind[vbeg[i]:vbeg[i+1]] nonzero indices for variable i
                    double *vval,  # vval[vbeg[i]:vbeg[i+1]] nonzero values for variable i
                    double *obj,
                    double *lb,
                    double *ub,
                    char *vtype,
                    const char **varnames
                    )

    int GRBaddvar(GRBmodel *model,
                  int nnz, int *vind, double *vval,
                  double obj, double lb, double ub,
                  char vtype, const char *varname)

    int GRBgetdblattrelement(GRBmodel *model,
                             const char *atttrname,
                             int element,
                             double *value)

    int GRBaddconstr(GRBmodel * model,
                       int nnz,
                       int *cind,
                       double *cval,
                       char sense,
                       double rhs,
                       const char *constrnames)

    int GRBupdatemodel(GRBmodel *model)

    # attributes
    int GRBgetintattr(GRBmodel *model,
                      const char *attrname,
                      int *value)

    int GRBgetdblattrarray(GRBmodel *model,
                           const char *attrname,
                           int start,
                           int length,
                           double *values)

    int GRBsetcharattrarray(GRBmodel *model,
                            const char *attrname,
                            int start,
                            int length,
                            char *values)