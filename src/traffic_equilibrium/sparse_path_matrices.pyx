# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from libc.stdio cimport printf, fprintf, stderr
from libc.stdlib cimport malloc, free
from .pathdb cimport PathDB, Cursor
from .solver cimport Problem, Result
from .igraph cimport igraph_vector_t, igraph_vector_append, igraph_vector_e_ptr
from .igraph_utils cimport vector_get, vector_len
from .linear_program cimport LPModel
from .gurobi cimport GRB_EQUAL, GRB_INFINITY, GRB_CONTINUOUS


#import gurobipy as gpy
import numpy as np
import time
from scipy.sparse import csc_matrix


DEF BYTES_PER_GiB = 1073742000.0

# data is stored as char
ctypedef char stored_t
# we want data in unsigned int
ctypedef unsigned int data_t

# TODO: turn into critical fleet size lp
def critical_fleet_size_model(Result result, PathDB db):
    cdef:
        igraph_vector_t* volume = result.problem.demand.volumes.vec
        size_t n_trips = vector_len(volume)
        igraph_vector_t* link_flow = result.flow.vec
        size_t n_links = vector_len(link_flow)
        int max_constraints = n_links + 1  # the max number of constraints that a path can participate in
        size_t n_paths = 0, n_edges = 0, nnz
        size_t *vbeg = NULL
        int *vind = NULL
        double *vval = NULL
        double one = 1
        size_t i, n
        size_t ratio = sizeof(data_t) / sizeof(stored_t)
        data_t *edges
        data_t edge_id, trip_id, j
        Cursor cursor = db.cursor()
        size_t log_every = 10000000
        double t0, duration
    printf("Initializing model...\n")
    model = LPModel()
    printf("Initializing constraints for %lu trips and %lu links...\n",
           n_trips, n_links)
    t0 = time.time()
    for i in range(n_trips):
        model.add_empty_constraint(GRB_EQUAL, vector_get(volume, i))
    for i in range(n_links):
        model.add_empty_constraint(GRB_EQUAL, vector_get(link_flow, i))
    duration = time.time() - t0
    model.update()
    printf("Created %i constraints in %g seconds.\n",
           model.n_constraints(), duration)
    printf("Populating constraints...\n")
    #vbeg = <size_t*> malloc(n_paths * sizeof(size_t))
    #vind = <int *> malloc(nnz * sizeof(int))
    #vval = <double *> malloc(nnz * sizeof(double))
    vind = <int *> malloc(max_constraints * sizeof(int))
    vval = <double *> malloc(max_constraints * sizeof(double))
    t0 = time.time()
    n_paths = 0
    cursor.reset()
    while cursor.is_valid():
        nnz = 0
        cursor.populate()
        # vbeg[n_paths] = nnz  # mark the beginning of a new column
        trip_id = (<data_t *> cursor.val_ptr)[0]
        # the first block of the constraints are trip flow constraints
        vind[nnz] = <int> trip_id
        vval[nnz] = one
        nnz += 1
        # the second block of the constraints are the link flow constraints
        # constraint index for link i is (i + n_trips)
        edges = <data_t*> cursor.key_ptr
        n = cursor.key_len / ratio
        for i in range(n):
            edge_id = <int> edges[i]
            vind[nnz] = edge_id + n_trips
            vval[nnz] = one
            nnz += 1
        model.add_variable(nnz, vind, vval,
                           0.0, 0.0, GRB_INFINITY,
                           GRB_CONTINUOUS, NULL)
        n_paths += 1
        if n_paths % log_every == 0:
            printf("Created %lu variables\n", n_paths)
            model.update()
        cursor.next()
    duration = time.time() - t0
    printf("Added %lu variables in %g seconds.\n",
           n_paths, duration)
    free(vind)
    free(vval)
    #model.add_variables(n_paths, nnz, vbeg, vind, vval, NULL, NULL, NULL)
    model.update()
    printf("Added %i variables\n", model.n_variables())
    printf("Returning model\n")
    return model
