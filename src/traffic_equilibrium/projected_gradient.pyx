# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,

from libc.stdlib cimport free, malloc, calloc
from libc.stdio cimport printf, setbuf, stdout
from libc.math cimport log2, INFINITY, fabs

from .leveldb cimport leveldb_writebatch_t, leveldb_writebatch_put
from .link_cost cimport LinkCost
from .timer cimport now
from .trips cimport OrgnDestDemand
from .vector cimport Vector
from .igraph cimport (
    igraph_t, igraph_vector_t, igraph_vector_ptr_t,
    igraph_vector_init, igraph_vector_destroy, igraph_vector_update,
    igraph_vector_fill,
    igraph_vector_scale,
    igraph_vector_resize,
    igraph_finally_func_t,
    igraph_vector_ptr_init,
    igraph_vector_ptr_reserve,
    igraph_vector_ptr_set_item_destructor, igraph_vector_ptr_destroy_all,
    igraph_vector_ptr_destroy,
    igraph_vector_all_e,
    igraph_vector_ptr_push_back,
    igraph_get_shortest_path_dijkstra,
    IGRAPH_OUT
)
from .igraph_utils cimport (
    vector_len, vector_get, vector_set, vector_ptr_get, vector_ptr_len,
    igraph_vector_dot,
)
from .solver cimport Problem
from .pathdb cimport write_path_to_batch, link_id_storage_t, trip_id_storage_t

import numpy as np
import os
import json
import datetime as dt

setbuf(stdout, NULL)


cdef inline void igraph_vector_e_add(igraph_vector_t *v, long int at, double plus) nogil:
    """Add an amount to a single element of a vector.
    
    Equivalent to: VECTOR(v)[at] += plus
    """
    vector_set(v, at,
               vector_get(v, at) + plus)

cdef int path_init(path_t *path) nogil:
    path.flow = 0.0
    path.cost = 0.0
    path.usable = True
    igraph_vector_init(&path.link_ids, 0)
    return 0

cdef int path_destroy(path_t *path) nogil:
    igraph_vector_destroy(&path.link_ids)
    return 0

cdef bint path_has_flow(path_t *path) nogil:
    return path.flow > 0.0

cdef bint path_is_congruent(path_t *path, igraph_vector_t *link_ids) nogil:
    return igraph_vector_all_e(&path.link_ids, link_ids)

cdef size_t path_get_link_id(path_t * path, size_t i) nogil:
    return <size_t> vector_get(&path.link_ids, i)

cdef int path_set_links(path_t* path, igraph_vector_t* links_ids) nogil:
    igraph_vector_update(&path.link_ids, links_ids)
    return 0

cdef int path_to_link_flow(path_t* path, igraph_vector_t * link_flow) nogil:
    cdef size_t i, n = vector_len(&path.link_ids)
    cdef long int eid
    for i in range(n):
        eid = path_get_link_id(path, i)
        igraph_vector_e_add(link_flow, eid, path.flow)
    return 0

cdef int path_update_link_flow(path_t* path, double new_flow,
                               igraph_vector_t* link_flow) nogil:
    """Update the flow on the path and adjust link flow accordingly"""
    cdef size_t i, n = vector_len(&path.link_ids)
    cdef long int eid
    cdef double diff = new_flow - path.flow
    for i in range(n):
        eid = path_get_link_id(path, i)
        igraph_vector_e_add(link_flow, eid, diff)
    path.flow = new_flow
    return 0

cdef int path_update_cost(path_t* path, igraph_vector_t* link_cost) nogil:
    cdef size_t i, eid, n = vector_len(&path.link_ids)
    path.cost = 0.0
    for i in range(n):
        eid = path_get_link_id(path, i)
        path.cost += vector_get(link_cost, eid)
    return 0

cdef void path_display(path_t *path, int level) nogil:
    cdef long int i
    printf("[flow: %g; cost: %g; ", path.flow, path.cost)
    if path.usable:
        printf("usable")
    else:
        printf("not usable")
    if level:
        printf("; links=")
        for i in range(vector_len(&path.link_ids)):
            printf(" %lu", <size_t> vector_get(&path.link_ids, i))
    printf("]")

cdef size_t path_number_of_links(path_t *path) nogil:
    return vector_len(&path.link_ids)

cdef void path_to_leveldb(path_t *path, size_t trip_id, leveldb_writebatch_t *writer) nogil:
    cdef trip_id_storage_t _trip_id = <trip_id_storage_t> trip_id
    if path_number_of_links(path):
        write_path_to_batch(writer, &path.link_ids, _trip_id)
    else:
        printf("ERROR: empty path found for trip %lu\n", trip_id)

cdef int trip_paths_init(trip_paths_t* trip_paths, size_t trip_id,
                         size_t source, size_t target, double demand) nogil:
    trip_paths.trip_id = trip_id
    trip_paths.source = source
    trip_paths.target = target
    trip_paths.demand = demand
    trip_paths.number_unused = 0
    trip_paths.paths = <igraph_vector_ptr_t*> malloc(sizeof(igraph_vector_ptr_t))
    igraph_vector_ptr_init(trip_paths.paths, 0)
    igraph_vector_ptr_set_item_destructor(trip_paths.paths,
                                          <igraph_finally_func_t*> path_destroy)
    return 0

cdef int trip_paths_destroy(trip_paths_t * trip_paths) nogil:
    if trip_paths.paths is not NULL:
        igraph_vector_ptr_destroy_all(trip_paths.paths)
        free(trip_paths.paths)
    return 0

cdef size_t trip_paths_number_of_paths(trip_paths_t * trip_paths) nogil:
    return <size_t> vector_ptr_len(trip_paths.paths)

cdef path_t* trip_paths_get_path(trip_paths_t * trip_paths, size_t i) nogil:
    return <path_t*> vector_ptr_get(trip_paths.paths, i)

cdef bint trip_paths_add(trip_paths_t * trip_paths, igraph_vector_t * link_ids,
                        double cost) nogil:
    cdef size_t i
    cdef path_t* path
    cdef double cost_margin = 1e-8
    if vector_len(link_ids) == 0:
        printf("WARNING: trying to add a path with no links on trip %lu!\n",
               trip_paths.trip_id)
        return False
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        # don't add the path if it is not better than a path by cost_margin
        if path.cost - cost_margin <= cost:
            # don't add the path if it is not actually strictly better
            return False
        if path_is_congruent(path, link_ids):
            path.cost = cost
            return False
    path = <path_t*> malloc(sizeof(path_t))
    path_init(path)
    path.cost = cost
    path_set_links(path, link_ids)
    igraph_vector_ptr_push_back(trip_paths.paths, path)
    return True

cdef size_t trip_paths_remove_unused(trip_paths_t * trip_paths,
                                     leveldb_writebatch_t* writer) nogil:
    cdef size_t i, removed = 0, n_paths = trip_paths_number_of_paths(trip_paths)
    cdef path_t* path
    cdef igraph_vector_ptr_t* reduced_paths = <igraph_vector_ptr_t*> malloc(sizeof(igraph_vector_ptr_t))
    if reduced_paths is NULL:
        printf("ERROR: failed to allocate memory for reduced_paths\n")
    igraph_vector_ptr_init(reduced_paths, 0)
    igraph_vector_ptr_reserve(reduced_paths, n_paths)
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        if path_has_flow(path):
            igraph_vector_ptr_push_back(reduced_paths, path)
        else:
            if writer is not NULL:
                path_to_leveldb(path, trip_paths.trip_id, writer)
            path_destroy(path)
            free(path)
            removed += 1
    igraph_vector_ptr_destroy(trip_paths.paths)
    free(trip_paths.paths)
    trip_paths.paths = reduced_paths
    trip_paths.number_unused = 0
    return removed

cdef void trip_paths_to_leveldb(trip_paths_t* trip_paths,
                                leveldb_writebatch_t *writer):
    cdef size_t i
    cdef path_t *path
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        path_to_leveldb(path, trip_paths.trip_id, writer)

cdef int trip_paths_to_link_flow(trip_paths_t * trip_paths,
                                 igraph_vector_t * link_flow) nogil:
    cdef size_t i
    cdef path_t * path
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        path_to_link_flow(path, link_flow)
    return 0

cdef int trip_paths_update_path_cost(trip_paths_t * trip_paths,
                                     igraph_vector_t * link_cost) nogil:
    cdef size_t i
    cdef path_t * path
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        path_update_cost(path, link_cost)
    return 0

cdef double trip_paths_fix_feasibility(trip_paths_t *trip_paths,
                                       double min_path_flow,
                                       igraph_vector_t *link_flow) nogil:
    cdef double extra, volume = 0.0, max_flow = 0.0, avg_flow
    cdef size_t i, max_i = 0, n_paths = trip_paths_number_of_paths(trip_paths)
    cdef path_t *path
    trip_paths.number_unused = 0
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        if path.flow <= min_path_flow or not path.usable:
            path_update_link_flow(path, 0.0, link_flow)
            path.usable = False
            trip_paths.number_unused += 1
        else:
            volume += path.flow
            if path.flow > max_flow:
                max_flow = path.flow
                max_i = i
    # extra is the amount we need to remove from the path flow
    extra = volume - trip_paths.demand
    if max_flow == 0.0:
        printf("ERROR: 0 of %lu paths for trip %lu has flow!\n",
               n_paths, trip_paths.trip_id)
        #trip_paths_display(trip_paths, 0)
        #printf("\n")
        # split the demand evenly over paths:
        avg_flow = trip_paths.demand / n_paths
        for i in range(n_paths):
            path = trip_paths_get_path(trip_paths, i)
            path_update_link_flow(path, avg_flow, link_flow)
            path.usable = True
        #trip_paths_display(trip_paths, 0)
        #printf("\n")
    else:
        path = trip_paths_get_path(trip_paths, max_i)
        if path.flow - extra <= 0.0:
            printf("Extra exceeds max flow on trip %lu: %g - %g <= 0.0 (volume = %g, demand = %g)\n",
                   trip_paths.trip_id, path.flow, extra, volume, trip_paths.demand)
            path_update_link_flow(path, 0.0, link_flow)
            path.usable = False
            trip_paths.number_unused += 1
        else:
            path.usable = True
            path_update_link_flow(path, path.flow - extra, link_flow)
    return fabs(extra)

cdef void trip_paths_display(trip_paths_t *trip_paths, int level) nogil:
    cdef size_t i
    cdef path_t *path
    printf("trip %lu ", trip_paths.trip_id)
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        path_display(path, level)

cdef inline double compute_step_size(double path_flow, double delta_path_flow) nogil:
    return fabs(path_flow / delta_path_flow)

cdef double trip_paths_search_direction(trip_paths_t* trip_paths,
                                        igraph_vector_t* direction) nogil:
    """Returns the maximum step size"""
    #cdef double average_cost = trip_paths_average_cost(trip_paths)
    cdef size_t i, count = 0, n = trip_paths_number_of_paths(trip_paths)
    igraph_vector_resize(direction, n)
    igraph_vector_fill(direction, 0)
    if n <= 1:
        return 0.0
    cdef double total_cost = 0.0
    cdef path_t* path
    for i in range(n):
        path = trip_paths_get_path(trip_paths, i)
        if path.cost < 0.0:
            printf("ERROR: negative path cost: %g\n", path.cost)
        if path.usable:
            count += 1
            total_cost += path.cost
        else:
            printf("ERROR: (trip_paths_search_direction) Found unusable path.\n")
    cdef double avg_cost = total_cost / (<double> count)
    cdef double max_step_size = INFINITY
    cdef double step_size, delta_path_flow
    cdef double net_direction = 0.0
    cdef double max_step_direction_error = 0.001 * trip_paths.demand
    cdef double cost_margin = 1e-8
    cdef bint within_cost_margin = True
    cdef igraph_vector_t path_flow, path_cost
    igraph_vector_init(&path_flow, n)
    igraph_vector_init(&path_cost, n)
    cdef bint usable_error = False
    for i in range(n):
        path = trip_paths_get_path(trip_paths, i)
        # delta path flow is negative when the path cost exceeds the average
        # and positive when the path cost is less than average
        if path.usable:
            #delta_path_flow = total_cost - (count * path.cost)
            delta_path_flow = avg_cost - path.cost
            if path.flow == 0.0 and delta_path_flow < 0.0:
                usable_error = True
        else:
            printf("ERROR: (trip_paths_search_direction) Found unusable path.\n")
            delta_path_flow = 0.0
        if fabs(delta_path_flow) >= cost_margin:
            within_cost_margin = False
        vector_set(direction, i, delta_path_flow)
        vector_set(&path_flow, i, path.flow)
        vector_set(&path_cost, i, path.cost)
        net_direction += delta_path_flow
        if delta_path_flow < 0.0:
            step_size = compute_step_size(path.flow, delta_path_flow)
            if step_size < max_step_size:
                max_step_size = step_size
    if usable_error and max_step_size:
        printf("ERROR: at least one path flow marked as usable with no flow and negative delta. trip id: %lu  max step size: %g direction: ",
               trip_paths.trip_id, max_step_size)
        igraph_vector_display(direction)
        printf(" flow: ")
        igraph_vector_display(&path_flow)
        printf(" cost: ")
        igraph_vector_display(&path_cost)
        printf("\n")
    if max_step_size < 0.0 or within_cost_margin:
        max_step_size = 0.0
    if fabs(net_direction * max_step_size) > max_step_direction_error:
        printf("ERROR: bad search direction (net direction = %g, average cost = %g) direction = ",
               net_direction, avg_cost)
        igraph_vector_display(direction)
        printf("path costs: ")
        igraph_vector_display(&path_cost)
        printf("path flow: ")
        igraph_vector_display(&path_flow)
        printf("\n")
        return 0.0
    igraph_vector_destroy(&path_flow)
    igraph_vector_destroy(&path_cost)
    if max_step_size < INFINITY:
        return max_step_size
    else:
        return 0.0

cdef inline void update_bound(size_t n_paths, trip_paths_t * trip_paths, double *ab) nogil:
    cdef size_t i
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        ab[i] = path.flow

cdef (bint, double) trip_paths_bisect(trip_paths_t *trip_paths,
                                      igraph_vector_t *link_flow,
                                      double *a, double *b, double dist) nogil:
    cdef bint negative_flow = False
    cdef double midpoint, width, total_width = 0.0
    cdef size_t i
    for i in range(trip_paths_number_of_paths(trip_paths)):
        path = trip_paths_get_path(trip_paths, i)
        width = b[i] - a[i]
        total_width += fabs(width)
        if dist == 0:
            midpoint = a[i]
        elif dist == 1:
            midpoint = b[i]
        else:
            midpoint = a[i] + dist * width
        if midpoint < 0.0:
            negative_flow = False
        path_update_link_flow(path, midpoint, link_flow)
    return negative_flow, total_width


cdef inline double max_of(double a, double b) nogil:
    return a if a >= b else b

cdef double trip_paths_line_search(trip_paths_t* trip_paths,
                                   LinkCost cost_fn,
                                   igraph_vector_t* direction,
                                   double max_step_size,
                                   igraph_vector_t* link_flow,
                                   igraph_vector_t* link_cost,
                                   double tolerance):
    cdef size_t i, j, n_paths = trip_paths_number_of_paths(trip_paths)
    if n_paths == 1 or max_step_size == 0.0:
        #printf("Skipping line search %lu paths, %g max step size, trip %lu\n",
        #       n_paths, max_step_size, trip_paths.trip_id)
        return 0.0
    cdef size_t eid, n_unused = 0, k = 0
    cdef path_t* path
    cdef double *a = <double*> calloc(n_paths, sizeof(double))
    cdef double *b = <double*> calloc(n_paths, sizeof(double))
    cdef double sigma, midpoint, step, width_tolerance, width = 0.0
    cdef bint negative_flow = False
    #cdef igraph_vector_t path_flow, path_cost
    #igraph_vector_init(&path_flow, n_paths)
    #igraph_vector_init(&path_cost, n_paths)
    cdef igraph_vector_t link_direction
    igraph_vector_init(&link_direction, vector_len(link_flow))
    #printf("extrema:\n")
    cdef double fa = 0.0, fb = 0.0
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        step = max_step_size * vector_get(direction, i)
        width += fabs(step)
        for j in range(path_number_of_links(path)):
            eid = path_get_link_id(path, j)
            igraph_vector_e_add(&link_direction, eid, step)
        if path.flow < 0.0:
            printf("ERROR: Found negative path flow on trip %lu: %g\n",
                   trip_paths.trip_id, path.flow)
        a[i] = path.flow
        b[i] = path.flow + step
    cdef double max_iterations = max_of(log2(width) - log2(tolerance), 24)
    fa = igraph_vector_dot(link_cost, &link_direction)
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        path_update_link_flow(path,
                              b[i],
                              link_flow)
    cost_fn.compute_link_cost(link_flow, link_cost)
    fb = igraph_vector_dot(link_cost, &link_direction)
    cdef bint done = False
    if fa * fb >= 0.0:
        # the interval does not contain a zero crossing in its interior
        # we want to return the endpoint with the smaller absolute derivative
        if fabs(fb) <= fabs(fa):
            # we've already updated the path, link flow, and link cost to the
            # b endpoint so just return
            sigma = fb
        else:
            # we need to reset path, link flow and link cost to the a endpoint
            for i in range(n_paths):
                path = trip_paths_get_path(trip_paths, i)
                path_update_link_flow(path, a[i], link_flow)
            cost_fn.compute_link_cost(link_flow, link_cost)
            sigma = fa
    else:
        #printf("\n")
        while not done:
            k += 1
            # Update path and link flow to midpoint
            negative_flow, width = trip_paths_bisect(
                trip_paths, link_flow,
                a, b, 0.5
            )
            if negative_flow:
                # the midpoint is not feasible because the the max step size was
                # too large. Move the upper bound to the mid point.
                if k < max_iterations:
                    update_bound(n_paths, trip_paths, b)
                else:
                    trip_paths_bisect(trip_paths, link_flow, a, b, 0)
                    cost_fn.compute_link_cost(link_flow, link_cost)
                    trip_paths_update_path_cost(trip_paths, link_cost)
                    sigma = fa
                    done = True
            else:
                # the upper bound is feasible, proceed as normal
                # compute link and path cost at midpoint
                cost_fn.compute_link_cost(link_flow, link_cost)
                trip_paths_update_path_cost(trip_paths, link_cost)
                # compute sigma
                sigma = igraph_vector_dot(link_cost, &link_direction)
                #printf("\t%lu: sigma=%g in [%g, %g] (delta=%0.12g) (width=%0.12g) (to endpoints [%g, %g])\n",
                #       k, sigma, fa, fb, fb - fa, width, sigma - fa, fb - sigma)
                if sigma < 0.0:
                    update_bound(n_paths, trip_paths, a)
                    fa = sigma
                else:
                    update_bound(n_paths, trip_paths, b)
                    fb = sigma
                done = fabs(sigma) < tolerance or k > max_iterations
    free(a)
    free(b)
    #igraph_vector_destroy(&path_flow)
    #igraph_vector_destroy(&path_cost)
    igraph_vector_destroy(&link_direction)
    return sigma

cdef class PathSet:
    def __cinit__(self, OrgnDestDemand demand, PathDB path_db):
        self.number_of_trips = <size_t> demand.number_of_trips()
        self.trip_paths = <trip_paths_t*> calloc(self.number_of_trips,
                                                 sizeof(trip_paths_t))
        self.path_db = path_db
        cdef size_t trip_id = 0
        cdef long int source_id, target_id
        cdef igraph_vector_t *source_targets
        cdef double volume
        for source_id in range(demand.number_of_sources()):
            source_targets = <igraph_vector_t*> vector_ptr_get(
                demand.targets.vec, source_id
            )
            for target_id in range(vector_len(source_targets)):
                volume = vector_get(demand.volumes.vec, trip_id)
                trip_paths_init(
                    &self.trip_paths[trip_id],
                    trip_id,
                    <size_t> vector_get(demand.sources.vec, source_id),
                    <size_t> vector_get(source_targets, target_id),
                    volume
                )
                trip_id += 1

    def __dealloc__(self):
        cdef size_t i
        for i in range(self.number_of_trips):
            trip_paths_destroy(&self.trip_paths[i])
        free(self.trip_paths)

    @staticmethod
    def create_from_problem(Problem problem, PathDB pathdb):
        return PathSet.__new__(PathSet, problem.demand, pathdb)

    cdef size_t number_of_paths(self) nogil:
        cdef size_t i, n_paths = 0
        for i in range(self.number_of_trips):
            n_paths += trip_paths_number_of_paths(&self.trip_paths[i])
        return n_paths

    cdef trip_paths_t* get_trip_paths(self, size_t trip_id) nogil:
        return &self.trip_paths[trip_id]

    cdef leveldb_writebatch_t* get_writer(self, size_t i) nogil:
        return self.path_db.writers[i]

    cdef void commit(self) nogil:
        self.path_db.commit()

    cdef void update_path_cost(self, igraph_vector_t *link_cost):
        cdef size_t trip_id
        cdef trip_paths_t *trip_paths
        for trip_id in range(self.number_of_trips):
            trip_paths = self.get_trip_paths(trip_id)
            trip_paths_update_path_cost(trip_paths, link_cost)

    cdef int to_link_flow(self, igraph_vector_t * link_flow) nogil:
        cdef size_t i
        for i in range(self.number_of_trips):
            trip_paths_to_link_flow(&self.trip_paths[i], link_flow)
        return 0

    def to_link_flow_numpy(self, long int nlinks):
        cdef igraph_vector_t link_flow
        cdef long int i
        igraph_vector_init(&link_flow, nlinks)
        self.to_link_flow(&link_flow)
        flow = np.zeros(nlinks)
        for i in range(nlinks):
            flow[i] = vector_get(&link_flow, i)
        igraph_vector_destroy(&link_flow)
        return flow

    cdef bint is_feasible(self, igraph_vector_t* volumes):
        cdef size_t trip_id, n_trips = self.number_of_trips
        cdef size_t i, n_paths, max_flow_i
        cdef trip_paths_t* trip_paths
        cdef path_t* path
        cdef double volume, max_flow = -1.0
        for trip_id in range(n_trips):
            trip_paths = self.get_trip_paths(trip_id)
            n_paths = trip_paths_number_of_paths(trip_paths)
            volume = 0
            for i in range(n_paths):
                path = trip_paths_get_path(trip_paths, i)
                if path.flow < 0.0:
                    printf("Found negative flow %0.10f (trip %lu)\n", path.flow, trip_id)
                    return False
                volume += path.flow
            if fabs(volume - vector_get(volumes, trip_id)) / volume > 1e-6:
                printf("Demand conservation constraint violated! (trip id %lu) %0.10f =/= %0.10f\n",
                       trip_id, volume, vector_get(volumes, trip_id))
                return False
        return True

    cdef void display(self):
        cdef size_t trip_id
        cdef trip_paths_t *trip_paths
        for trip_id in range(self.number_of_trips):
            trip_paths_display(self.get_trip_paths(trip_id), 1)

    cdef void save_paths(self):
        cdef size_t trip_id
        cdef trip_paths_t *trip_paths
        for trip_id in range(self.number_of_trips):
            trip_paths_to_leveldb(self.get_trip_paths(trip_id),
                                  self.get_writer(0))
        self.commit()

cdef void igraph_vector_display(igraph_vector_t *v) nogil:
    cdef long int i
    for i in range(vector_len(v)):
        printf("%g ", vector_get(v, i))

cdef bint trip_paths_add_shortest_path(
        trip_paths_t *trip_paths,
        igraph_t *graph,
        igraph_vector_t *link_cost,
        igraph_vector_t *path_links
) nogil:
    cdef bint is_new_path
    # ensure that trip paths have up-to-date costs
    trip_paths_update_path_cost(trip_paths, link_cost)
    # compute the shortest path
    cdef igraph_vector_t nodes
    igraph_vector_init(&nodes, 0)
    cdef int errno = igraph_get_shortest_path_dijkstra(
        graph,
        &nodes, #NULL,
        path_links,
        <int> trip_paths.source,
        <int> trip_paths.target,
        link_cost,
        IGRAPH_OUT
    )
    if errno:
        printf("ERROR: shortest paths returned non-zero exit code: %d\n",
               errno)
        return False
    # compute the cost of the shortest path
    cdef long int i, eid, n_links = vector_len(path_links)
    cdef double cost = 0.0
    if n_links == 0:
        printf("WARNING: path with no links found for trip %lu (%lu->%lu); nodes found: ",
               trip_paths.trip_id, trip_paths.source, trip_paths.target)
        igraph_vector_display(&nodes)
        printf("\n")
    for i in range(n_links):
        eid = <long int> vector_get(path_links, i)
        cost += vector_get(link_cost, eid)
    # add the path to the path set
    is_new_path = trip_paths_add(trip_paths, path_links, cost)
    igraph_vector_destroy(&nodes)
    return is_new_path


cdef void costs_reset(costs_t *costs) nogil:
    costs.total_flow = 0.0
    costs.total_cost = 0.0
    costs.total_shortest_paths_cost = 0.0
    costs.maximum_excess_cost = 0.0

cdef void costs_display(costs_t *costs) nogil:
    printf("total flow %g; total cost: %g; total cost on shortest paths: %g; maximum excess cost (so far) %g\n",
           costs.total_flow, costs.total_cost, costs.total_shortest_paths_cost, costs.maximum_excess_cost)

cdef void costs_update(costs_t *costs,
                       double total_cost, double total_flow,
                       double max_cost, double min_cost) nogil:
    cdef double max_excess = max_cost - min_cost
    costs.total_flow += total_flow
    costs.total_cost += total_cost
    costs.total_shortest_paths_cost += total_flow * min_cost
    if max_excess > costs.maximum_excess_cost:
        costs.maximum_excess_cost = max_excess

cdef void metrics_compute_from_costs(metrics_t *metrics, costs_t *costs) nogil:
    metrics.relative_gap = (
        (costs.total_cost - costs.total_shortest_paths_cost)
        / costs.total_cost
    )
    metrics.average_excess_cost = (
        (costs.total_cost - costs.total_shortest_paths_cost)
        / costs.total_flow
    )
    metrics.maximum_excess_cost = costs.maximum_excess_cost

cdef void trip_paths_append_costs(trip_paths_t *trip_paths,
                                  costs_t *costs) nogil:
    cdef size_t i, n_paths = trip_paths_number_of_paths(trip_paths)
    cdef path_t *path
    cdef double min_cost = INFINITY
    cdef double max_cost = 0.0
    cdef double total_cost = 0.0
    cdef double max_excess_cost
    for i in range(n_paths):
        path = trip_paths_get_path(trip_paths, i)
        total_cost += path.flow * path.cost
        if path.cost < min_cost:
            min_cost = path.cost
        if path.cost > max_cost:
            max_cost = path.cost
    costs_update(costs, total_cost, trip_paths.demand,
                 max_cost, min_cost)

cdef int trip_paths_load_initial(trip_paths_t *trip_paths,
                                  igraph_t *graph,
                                  igraph_vector_t *link_flow,
                                  igraph_vector_t *link_cost,
                                  igraph_vector_t *path_links,
                                  ) except -1:
    cdef path_t *path
    cdef size_t n_paths = trip_paths_number_of_paths(trip_paths)
    if n_paths:
        raise Exception(
            "ERROR: There are already paths for trip %lu; cannot load initial path."
            % trip_paths.trip_id
        )
    trip_paths_add_shortest_path(
        trip_paths,
        graph,
        link_cost,
        path_links
    )
    n_paths = trip_paths_number_of_paths(trip_paths)
    if n_paths == 0:
        raise Exception(
            "No paths added for trip %lu; cannot load initial path."
        )
    path = trip_paths_get_path(trip_paths, 0)
    path_update_link_flow(path, trip_paths.demand, link_flow)
    return 0

cdef void trip_paths_update_path_flow(
        Problem problem,
        trip_paths_t *trip_paths,
        igraph_vector_t *link_flow,
        igraph_vector_t *link_cost,
        igraph_vector_t *direction,
        igraph_vector_t *path_links,
        leveldb_writebatch_t *writer,
        costs_t *costs,
        double tolerance,
        double minimum_path_flow,
):
    cdef igraph_vector_t *volumes = problem.demand.volumes.vec
    cdef LinkCost cost_fn = problem.cost_fn
    cdef igraph_t *graph = problem.network.graph
    cdef double max_step_size, best_trip_tt, error = 0.0
    cdef bint new_path
    cdef int i
    for i in range(24):
        # compute direction and max step size
        max_step_size = trip_paths_search_direction(trip_paths, direction)
        # update path flow, link flow, and link cost
        trip_paths_line_search(
            trip_paths,
            cost_fn,
            direction,
            max_step_size,
            link_flow,
            link_cost,
            tolerance
        )
        if trip_paths_number_of_paths(trip_paths):
            error += trip_paths_fix_feasibility(
                trip_paths,
                minimum_path_flow,
                link_flow
            )
        # remove unused paths
        if trip_paths.number_unused:
            trip_paths_remove_unused(trip_paths, writer)
        cost_fn.compute_link_cost(link_flow, link_cost)
        # add new shortest paths
        new_path = trip_paths_add_shortest_path(
            trip_paths,
            graph,
            link_cost,
            path_links
        )
        if not new_path:
            break
    # compute metrics
    trip_paths_append_costs(trip_paths, costs)

cdef int initial_path_flow(
        Problem problem,
        PathSet path_set,
        igraph_vector_t *link_flow,
        igraph_vector_t *link_cost,
        igraph_vector_t *path_links,
) except -1:
    cdef size_t trip_id, n_trips = path_set.number_of_trips
    cdef igraph_t *graph = problem.network.graph
    cdef trip_paths_t *trip_paths
    problem.cost_fn.compute_link_cost(link_flow, link_cost)
    for trip_id in range(n_trips):
        trip_paths = path_set.get_trip_paths(trip_id)
        trip_paths_load_initial(
            trip_paths,
            graph,
            link_flow,
            link_cost,
            path_links,
        )
    problem.cost_fn.compute_link_cost(link_flow, link_cost)
    return 0


cdef size_t update_path_flow(Problem problem, PathSet path_set,
                           igraph_vector_t *link_cost,
                           igraph_vector_t *link_flow,
                           igraph_vector_t *direction,
                           igraph_vector_t *path_links,
                           double tolerance,
                           metrics_t *metrics,
                           leveldb_writebatch_t *writer):
    cdef double minimum_path_flow = 0.0
    cdef size_t n_paths = 0, trip_id = 0, n_trips = path_set.number_of_trips
    cdef costs_t costs
    costs_reset(&costs)
    for trip_id in range(n_trips):
        trip_paths = path_set.get_trip_paths(trip_id)
        trip_paths_update_path_flow(
            problem,
            trip_paths,
            link_flow,
            link_cost,
            direction,
            path_links,
            writer,
            &costs,
            tolerance,
            minimum_path_flow
        )
        n_paths += trip_paths_number_of_paths(trip_paths)
    metrics_compute_from_costs(metrics, &costs)
    return n_paths

cdef void metrics_clear(metrics_t * metrics) nogil:
    metrics.relative_gap = 0.0
    metrics.average_excess_cost = 0.0
    metrics.maximum_excess_cost = 0.0

cdef dict metrics_to_dict(metrics_t *metrics):
    return {
        'relative_gap': metrics.relative_gap,
        'maximum_excess_cost': metrics.maximum_excess_cost,
        'average_excess_cost': metrics.average_excess_cost,
        'iterations': metrics.iterations,
    }

cdef void metrics_update_best(metrics_t *best_metrics, metrics_t *metrics) nogil:
    if metrics.relative_gap < best_metrics.relative_gap:
        best_metrics.relative_gap = metrics.relative_gap
        best_metrics.iterations = metrics.iterations
    if metrics.average_excess_cost < best_metrics.average_excess_cost:
        best_metrics.average_excess_cost = metrics.average_excess_cost
        best_metrics.iterations = metrics.iterations
    if metrics.maximum_excess_cost < best_metrics.maximum_excess_cost:
        best_metrics.maximum_excess_cost = metrics.maximum_excess_cost
        best_metrics.iterations = metrics.iterations

cdef bint is_converged(metrics_t *tolerance, metrics_t *values) nogil:
    """Determines whether or not the algorithm has converged"""
    if values.relative_gap < tolerance.relative_gap:
        printf("Terminated via relative gap tolerance: %g < %g.\n",
               values.relative_gap, tolerance.relative_gap)
        return True
    if values.average_excess_cost < tolerance.average_excess_cost:
        printf("Terminated via average excess tolerance: %g < %g.\n",
               values.average_excess_cost, tolerance.average_excess_cost)
        return True
    if values.maximum_excess_cost < tolerance.maximum_excess_cost:
        printf("Terminated via maximum excess tolerance: %g < %g.\n",
               values.maximum_excess_cost, tolerance.maximum_excess_cost)
        return True
    if values.iterations > tolerance.iterations:
        printf("Terminated via iteration tolerance: %lu > %lu.\n",
               values.iterations, tolerance.iterations)
        return True
    return False

cdef class ProjectedGradientSettings:
    def __cinit__(self,
                   double rgap_tol,
                   double mec_tol,
                   double aec_tol,
                   size_t max_iterations,
                   double line_search_tolerance,
                   size_t report_every
                   ):
        self.tolerance.relative_gap = rgap_tol
        self.tolerance.maximum_excess_cost = mec_tol
        self.tolerance.average_excess_cost = aec_tol
        self.tolerance.iterations = max_iterations
        self.line_search_tolerance = line_search_tolerance
        self.report_every = report_every

    def to_dict(self):
        return {
            'tolerance': metrics_to_dict(&self.tolerance),
            'line_search_tolerance': self.line_search_tolerance,
            'report_every': self.report_every
        }

    @staticmethod
    def from_dict(dict data):
        return ProjectedGradientSettings(
            data['tolerance']['relative_gap'],
            data['tolerance']['maximum_excess_cost'],
            data['tolerance']['average_excess_cost'],
            data['tolerance']['iterations'],
            data['line_search_tolerance'],
            data['report_every']
        )


cdef class ProjectedGradientResult:
    def __cinit__(self, Problem problem, PathSet path_set,
                  ProjectedGradientSettings settings):
        self.metrics.relative_gap = INFINITY
        self.metrics.average_excess_cost = INFINITY
        self.metrics.maximum_excess_cost = INFINITY
        self.metrics.iterations = 0

        self.best_metrics.relative_gap = INFINITY
        self.best_metrics.average_excess_cost = INFINITY
        self.best_metrics.maximum_excess_cost = INFINITY
        self.best_metrics.iterations = 0

        self.problem = problem
        self.path_set = path_set
        self.paths = self.path_set.path_db
        self.flow = Vector.zeros(self.problem.network.number_of_links())
        self.settings = settings

    def save(self, name):
        timestamp = dt.datetime.utcnow().isoformat(timespec='seconds')
        dirname = os.path.join(name, f"results-{timestamp}")
        os.makedirs(dirname)
        print(f"Saving to {dirname}")
        self.path_set.save_paths()
        self.problem.save(dirname)
        data = {
            'metrics': metrics_to_dict(&self.metrics),
            'settings': self.settings.to_dict(),
            'pathdb': self.path_set.path_db.name
        }
        with open(os.path.join(dirname, 'metadata.json'), 'w') as fp:
            json.dump(data, fp)
        cdef long int nlinks = self.problem.network.number_of_links()
        np.savez(os.path.join(dirname, "arrays"),
                 link_flow=self.flow.to_array()
        )

    @staticmethod
    def load(str dirname):
        problem = Problem.load(dirname)
        with open(os.path.join(dirname, 'metadata.json')) as fp:
            data = json.load(fp)
        pathdb = PathDB(data['pathdb'])
        pathset = PathSet.create_from_problem(problem, pathdb)
        settings = ProjectedGradientSettings.from_dict(data['settings'])
        result = ProjectedGradientResult(
            problem,
            pathset,
            settings
        )
        arrays = np.load(os.path.join(dirname, "arrays.npz"))
        result.flow = Vector.copy_of(arrays['link_flow'])
        result.metrics.iterations = data['metrics']['iterations']
        result.metrics.maximum_excess_cost = data['metrics']['maximum_excess_cost']
        result.metrics.average_excess_cost = data['metrics']['average_excess_cost']
        return result

    cdef void print_report(self, size_t n_paths, double t0):
        cdef double duration = now() - t0
        printf("%lu: rgap=%g (%g), aec=%g (%g), mec=%g (%g) -- total paths: %lu, timing %g it/s, time elapsed: %g\n",
               self.metrics.iterations,
               self.metrics.relative_gap, self.best_metrics.relative_gap,
               self.metrics.average_excess_cost, self.best_metrics.average_excess_cost,
               self.metrics.maximum_excess_cost, self.best_metrics.maximum_excess_cost,
               n_paths, self.metrics.iterations / duration, duration
               )

    cdef void update_path_costs(self, igraph_vector_t* link_flow,
                                igraph_vector_t* link_cost):
        self.problem.cost_fn.compute_link_cost(link_flow, link_cost)
        self.path_set.update_path_cost(link_cost)

    cdef bint is_feasible(self):
        return self.path_set.is_feasible(self.problem.demand.volumes.vec)

    cpdef void solve_via_projected_gradient(self):
        cdef size_t n_paths, number_of_links = self.problem.network.number_of_links()
        cdef igraph_vector_t link_cost, link_flow, direction, path_links

        igraph_vector_init(&link_cost, number_of_links)
        igraph_vector_init(&link_flow, number_of_links)
        igraph_vector_init(&path_links, number_of_links)
        igraph_vector_init(&direction, number_of_links)

        cdef bint converged = False
        cdef double t0 = now()
        initial_path_flow(self.problem,
                          self.path_set,
                          &link_flow,
                          &link_cost,
                          &path_links)
        printf("Path Set contains %lu paths.\n", self.path_set.number_of_paths())
        assert self.is_feasible()
        self.metrics.iterations = 0
        while not converged:
            n_paths = update_path_flow(
                self.problem,
                self.path_set,
                &link_cost,
                &link_flow,
                &direction,
                &path_links,
                self.settings.line_search_tolerance,
                &self.metrics,
                self.path_set.get_writer(0)
            )
            metrics_update_best(&self.best_metrics, &self.metrics)
            if self.metrics.iterations % self.settings.report_every == 0:
                self.print_report(n_paths, t0)
            self.metrics.iterations += 1
            converged = is_converged(&self.settings.tolerance, &self.metrics)
            # ensure the link flow doesn't drift over iterations
            igraph_vector_fill(&link_flow, 0.0)
            self.path_set.to_link_flow(&link_flow)
            self.update_path_costs(&link_flow, &link_cost)
            # commit anything in the writebatch (removed paths)
            self.path_set.commit()
        printf("\nDone.\n")
        self.print_report(n_paths, t0)
        igraph_vector_update(self.flow.vec, &link_flow)
        # TODO: destroy vectors
