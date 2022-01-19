# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from collections import namedtuple

from libc.stdio cimport printf, setbuf, stdout
from libc.stdint cimport int32_t, int64_t, uint8_t
cimport numpy as np
from cython cimport view

from .igraph cimport (igraph_vector_t, igraph_vector_update,
    igraph_vector_mul, igraph_vector_add, igraph_vector_init,
    igraph_vector_destroy)
from .igraph_utils cimport vector_get, vector_len
from .pathdb cimport read_path_from_cursor, trip_id_storage_t, link_id_storage_t
from .timer cimport now
from .vector cimport Vector, PointerVector
from .pathdb cimport PathDB, Cursor
from .solver cimport Result, Problem
from .link_cost cimport LinkCost
from .network_loading cimport shortest_paths_assignment

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csc_matrix, hstack
import heapq
import matplotlib.pyplot as plt

setbuf(stdout, NULL)

def check_fleet_paths(Problem problem, PathDB paths,
                      Vector link_flow, Vector fleet_link_flow, LinkCost link_cost_fn,
                      link_path, fleet_marginal_trip_cost, tolerance=1e-2, cfs_so=True):
    # compute fleet marginal link cost
    cdef size_t n_links = vector_len(link_flow.vec)
    cdef Vector tmp_flow = Vector.zeros(n_links)
    cdef Vector link_cost = Vector.zeros(n_links)
    cdef size_t n_trips = problem.demand.number_of_trips()
    cdef Vector trip_cost = Vector.zeros(n_trips)
    link_cost_fn.compute_link_cost(link_flow.vec, link_cost.vec)
    cdef Vector link_cost_gradient = Vector.zeros(n_links)
    link_cost_fn.gradient(link_flow.vec, link_cost_gradient.vec)
    cdef Vector fleet_marginal_link_cost = Vector.zeros(n_links)
    compute_fleet_link_marginal_cost(fleet_marginal_link_cost.vec,
                                     link_cost.vec,
                                     fleet_link_flow.vec,
                                     link_cost_gradient.vec)
    cdef Vector cost_for_spa
    if cfs_so:
        cost_for_spa = fleet_marginal_link_cost
    else:
        cost_for_spa = link_cost
    cdef PathDB _paths = PathDB("paths.tmp.db")
    shortest_paths_assignment(problem.network,
                              cost_for_spa,
                              problem.demand,
                              tmp_flow,
                              _paths,
                              trip_cost,
                              )
    _paths.commit()
    # Add paths to real db as well
    shortest_paths_assignment(problem.network,
                              cost_for_spa,
                              problem.demand,
                              tmp_flow,
                              paths,
                              trip_cost,
                              )
    paths.commit()
    trip_cost_np = trip_cost.to_array()
    #link_path_norm = link_path.sum(0)
    link_vector = np.zeros(n_links, dtype=np.int32)
    cdef int[:] _link_vector = link_vector
    cdef size_t i, m, new_paths = 0
    cdef unsigned int eid
    cdef list lp_index = []
    cdef list lp_indptr = [0]
    cdef list tp_index = []
    cdef list tp_indptr = [0]
    for k, v in _paths.items():
        for i in range(n_links):
            _link_vector[i] = 0
        m = len(k)
        for i in range(m):
            eid = k[i]
            _link_vector[eid] = 1
        # finds paths in link_path that include all links in link_vector
        # contains is True if path is a sub path of one in link_path
        contains = (link_vector @ link_path) == m
        # finds paths in link_path that include no links that are not in link_vector
        # contained is True if a path in link_path is a sub path of path
        contained = ((1 - link_vector) @ link_path) == 0
        idx = contains & contained
        if not idx.any():
            new_paths += 1
            for i in range(m):
                eid = k[i]
                lp_index.append(eid)
            lp_indptr.append(len(lp_index))
            tp_index.append(v[0])
            tp_indptr.append(len(tp_index))
    assert new_paths <= n_trips
    lp_data = np.ones(len(lp_index), dtype=np.int)
    tp_data = np.ones(len(tp_index), dtype=np.int)
    lp = csc_matrix((lp_data, lp_index, lp_indptr), shape=(n_links, new_paths))
    tp = csc_matrix((tp_data, tp_index, tp_indptr), shape=(n_trips, new_paths))
    _paths.destroy_db()
    return new_paths, lp, tp


cdef void compute_fleet_link_marginal_cost(igraph_vector_t *fmlc,
                                   igraph_vector_t *lc,
                                   igraph_vector_t *flf,
                                   igraph_vector_t *grad):
    igraph_vector_update(fmlc, flf)  # fmlf = flf
    igraph_vector_mul(fmlc, grad)  # fmlf = flf * grad
    igraph_vector_add(fmlc, lc)  # fmlf = lc + flf * grad


class CriticalFleetSizeProblem:
    def __init__(self, problem, fleet_path_flow, user_path_flow, fleet_link_flow, user_link_flow, aggregate_link_flow,
                 fleet_marginal_path_cost, fleet_paths=None):
        self.problem = problem
        self.fleet_path_flow = fleet_path_flow
        self.user_path_flow = user_path_flow
        self.fleet_link_flow = fleet_link_flow
        self.user_link_flow = user_link_flow
        self.aggregate_link_flow = aggregate_link_flow
        self.fleet_marginal_path_cost = fleet_marginal_path_cost
        self.fleet_paths = fleet_paths

    def solve(self, *args, **kwargs):
        res = self.problem.solve(*args, **kwargs)
        return res

    def is_feasible(self, so_path_flow):
        self.user_path_flow.project_and_assign(np.zeros_like(so_path_flow))
        self.fleet_path_flow.project_and_assign(so_path_flow)
        violations = []
        for i, constraint in enumerate(self.problem.constraints):
            is_feasible = constraint.value()
            if not is_feasible:
                violations.append(i)
        return violations

    def is_feasible_ue(self, ue_path_flow):
        self.fleet_path_flow.project_and_assign(np.zeros_like(ue_path_flow))
        self.user_path_flow.project_and_assign(ue_path_flow)
        violations = []
        for i, constraint in enumerate(self.problem.constraints):
            is_feasible = constraint.value()
            print(i, is_feasible)
            if not is_feasible:
                print("adding constraint to violations")
                violations.append(i)
        return violations

    def total_volume(self):
        return cp.sum(
            self.fleet_path_flow + self.user_path_flow
        ).value

    def fleet_fraction(self):
        fleet_volume = np.sum(self.fleet_path_flow.value)
        return cp.sum(self.fleet_path_flow).value / self.total_volume()

    def link_flow_error(self):
        return cp.norm2(
            self.user_link_flow
            + self.fleet_link_flow
            - self.aggregate_link_flow
        ).value / self.total_volume()

    @property
    def fleet_marginal_trip_cost(self):
        return self.problem.var_dict["fleet_marginal_cost"]


cdef validate_vector(Vector vec):
    a = vec.to_array()
    if np.isfinite(a).all():
        return True
    else:
        if np.isinf(a).any():
            print("contains infinity")
        if np.isnan(a).any():
            print("contains nan")
        return False


def prepare_critical_fleet_size_problem(Result result,
                                        LinkCost lpf, LinkCost mlpf,
                                        double epsilon_user,
                                        double epsilon_fleet,
                                        bint use_all_paths=False,
                                        unsigned int max_paths_per_trip=100):
    return prepare_critical_fleet_size(result.problem,
                                       result.paths,
                                       result.prev_flow,
                                       result.flow,
                                       lpf, mlpf,
                                       epsilon_user,
                                       epsilon_fleet,
                                       use_all_paths,
                                       max_paths_per_trip)

def prepare_critical_fleet_size(Problem problem,
                                PathDB paths,
                                Vector prev_flow,
                                Vector link_flow,
                                LinkCost lpf, LinkCost mlpf,
                                double epsilon_user,
                                double epsilon_fleet,
                                bint use_all_paths=False,
                                unsigned int max_paths_per_trip=100,
                                bint cfs_so=True):
    cdef bint recompute_paths = False
    cdef unsigned int n_links = problem.network.number_of_links()
    cdef unsigned int n_trips = problem.demand.number_of_trips()
    printf("Found %u trips.\n", n_trips)
    cdef Vector link_cost = Vector.zeros(n_links)
    cdef Vector link_marginal_cost = Vector.zeros(n_links)
    cdef Vector link_cost_gradient = Vector.zeros(n_links)
    cdef Vector trip_cost = Vector.zeros(n_trips)
    cdef Vector trip_marginal_cost = Vector.zeros(n_trips)

    # populate link cost and link marginal cost
    lpf.compute_link_cost(prev_flow.vec, link_cost.vec)
    lpf.gradient(prev_flow.vec, link_cost_gradient.vec)
    mlpf.compute_link_cost(prev_flow.vec, link_marginal_cost.vec)

    assert validate_vector(link_cost), "link cost is infinte or nan"
    assert validate_vector(link_cost_gradient), "link cost gradient is infinite or nan"
    assert validate_vector(link_marginal_cost), "link marginal cost is infinite or nan"

    # populate trip cost
    cdef Vector tmp_flow = Vector.zeros(n_links)
    shortest_paths_assignment(
        problem.network,
        link_cost,
        problem.demand,
        tmp_flow,
        paths,
        trip_cost,
    )
    paths.commit()
    shortest_paths_assignment(
        problem.network,
        link_marginal_cost,
        problem.demand,
        tmp_flow,
        paths,
        trip_marginal_cost
    )
    paths.commit()

    link_path, trip_path, fleet_paths, user_paths = get_epsilon_paths(
        paths,
        link_cost, link_marginal_cost,
        trip_cost, trip_marginal_cost,
        epsilon_user, epsilon_fleet,
        use_all_paths,
        max_paths_per_trip,
        cfs_so
    )

    link_flow_est, path_flow_est, link_flow_epsilon = get_projected_link_flow(link_flow,
                                            problem.demand.volumes,
                                            lpf,
                                            trip_path, link_path, fleet_paths)
    return (
        link_path, trip_path,
        fleet_paths, user_paths,
        link_flow.to_array(),
        link_flow_est,
        link_flow_epsilon,
        path_flow_est,
        link_cost_gradient, link_marginal_cost,
        link_cost,
        trip_cost, trip_marginal_cost
    )

cdef double minimum_cost_threshold_(double cost, double epsilon):
    """Additive cost threshold"""
    return cost + epsilon

cdef double minimum_cost_threshold(double cost, double epsilon):
    """Relative gap based cost threshold"""
    return cost * (epsilon + 1)

TripPathItem = namedtuple('TripPathItem',
                          ['neg_cost', 'path_id', 'trip_id', 'links'])


cdef class UsablePaths:
    cdef unsigned int max_items
    cdef bint unlimited
    cdef object trip_paths_index
    cdef object path_ids

    def __cinit__(self, unsigned int n_trips, unsigned int max_items):
        if max_items == 0:
            self.unlimited = True
        else:
            self.unlimited = False
        self.max_items = max_items
        self.path_ids = {}
        cdef unsigned int i
        self.trip_paths_index = [
            list() for i in range(n_trips)
        ]

    @staticmethod
    def min_cost(list paths):
        return -max(paths).neg_cost

    def threshold(self, double epsilon):
        cdef unsigned int trip_id, n = len(self.trip_paths_index)
        cdef list paths
        cdef double min_cost
        for trip_id in range(n):
            paths = self.trip_paths_index[trip_id]
            min_cost = self.min_cost(paths)
            while -paths[0].neg_cost > minimum_cost_threshold(min_cost, epsilon):
                item = heapq.heappop(paths)
                del self.path_ids[item.path_id]

    cdef append(self, trip_id_storage_t trip, double cost, size_t path_id, igraph_vector_t *path):
        cdef list paths = self.trip_paths_index[trip]
        cdef double neg_cost = -cost
        cdef long int i, n = vector_len(path)
        self.path_ids[path_id] = n
        links = np.zeros(n, np.uint)
        cdef size_t[:] _links = links
        for i in range(n):
            _links[i] = <unsigned int> vector_get(path, i)
        if self.unlimited or (len(paths) < self.max_items):
            heapq.heappush(paths, TripPathItem(neg_cost, path_id, trip, links))
        else:
            item = heapq.heappushpop(paths, TripPathItem(neg_cost, path_id, trip, links))
            del self.path_ids[item.path_id]

    def number_of_paths(self):
        return len(self.path_ids)

    def get_path_ids(self):
        return self.path_ids.keys()

    def path_iter(self):
        for paths in self.trip_paths_index:
            for path in paths:
                yield path


def get_epsilon_paths(PathDB paths,
                      Vector link_cost, Vector link_marginal_cost,
                      Vector trip_cost, Vector trip_marginal_cost,
                      double epsilon_user, double epsilon_fleet,
                      bint use_all_paths=False,
                      unsigned int max_paths=100,
                      bint cfs_so=True):
    if use_all_paths:
        max_paths=0
    return get_epsilon_paths_mem(
        paths,
        link_cost, link_marginal_cost,
        trip_cost, trip_marginal_cost,
        epsilon_user, epsilon_fleet,
        use_all_paths, max_paths, cfs_so)


def get_epsilon_paths_mem(PathDB paths,
                      Vector link_cost, Vector link_marginal_cost,
                      Vector trip_cost, Vector trip_marginal_cost,
                      double epsilon_user, double epsilon_fleet,
                      bint use_all_paths=False,
                      unsigned int max_paths=100,
                      bint cfs_so=True):
    cdef Cursor cursor = paths.cursor()
    cdef long int i, path_len, n_links = vector_len(link_cost.vec), n_trips = vector_len(trip_cost.vec)
    cdef UsablePaths user_paths = UsablePaths(n_trips, max_paths)
    cdef UsablePaths fleet_paths = UsablePaths(n_trips, max_paths)
    cdef UsablePaths all_paths = UsablePaths(n_trips, 0)

    cdef size_t path_id = 0, nnz = 0, n_recovered_paths
    cdef double cost, marginal_cost, fleet_cost
    cdef trip_id_storage_t trip_id
    cdef link_id_storage_t eid
    cdef igraph_vector_t path
    igraph_vector_init(&path, 0)
    cdef double t0 = now()
    cursor.reset()
    while cursor.is_valid():
        cursor.populate()
        trip_id = read_path_from_cursor(cursor, &path)
        path_len = vector_len(&path)
        if path_len:
            cost = 0.0
            marginal_cost = 0.0
            for i in range(path_len):
                eid = <link_id_storage_t> vector_get(&path, i)
                cost += vector_get(link_cost.vec, eid)
                marginal_cost += vector_get(link_marginal_cost.vec, eid)
            user_paths.append(trip_id, cost, path_id, &path)
            if cfs_so:
                fleet_cost = marginal_cost
            else:
                fleet_cost = cost
            fleet_paths.append(trip_id, fleet_cost, path_id, &path)
            if use_all_paths:
                all_paths.append(trip_id, path_id, path_id, &path)
        # prepare for next iteration
        path_id += 1
        cursor.next()
    user_paths.threshold(epsilon_user)
    fleet_paths.threshold(epsilon_fleet)
    cdef set usable_path_ids = set()
    cdef dict user_path_ids = user_paths.path_ids
    cdef dict fleet_path_ids = fleet_paths.path_ids
    cdef dict all_path_ids = all_paths.path_ids
    usable_path_ids.update(user_path_ids)
    usable_path_ids.update(fleet_path_ids)
    usable_path_ids.update(all_path_ids)
    for pid in usable_path_ids:
        nnz += fleet_path_ids.get(pid) or user_path_ids.get(pid) or all_path_ids.get(pid)
    n_recovered_paths = len(usable_path_ids)
    printf("Found %lu paths in %g seconds; %lu usable (%lu fleet | %lu user)\n",
           path_id, now() - t0,
           n_recovered_paths, len(fleet_path_ids), len(user_path_ids)
           )
    lp_index = np.zeros(nnz, dtype=np.int64)
    lp_indptr = np.zeros(n_recovered_paths + 1, dtype=np.int64)
    cdef int64_t[:] _lp_index = lp_index
    cdef int64_t[:] _lp_indptr = lp_indptr
    tp_index = np.zeros(n_recovered_paths, dtype=np.int32)
    tp_indptr = np.zeros(n_recovered_paths + 1, dtype=np.int32)
    cdef int32_t[:] _tp_index = tp_index
    cdef int32_t[:] _tp_indptr = tp_indptr
    fleet_paths_mask = np.zeros(n_recovered_paths, dtype=np.bool)
    user_paths_mask = np.zeros(n_recovered_paths, dtype=np.bool)
    cdef uint8_t[:] _fleet_paths_mask = fleet_paths_mask
    cdef uint8_t[:] _user_paths_mask = user_paths_mask

    printf("Populating sparse matrices...")
    path_id = 0
    nnz = 0
    t0 = now()
    for item in fleet_paths.path_iter():
        pid = item.path_id
        usable_path_ids.remove(pid)
        _fleet_paths_mask[path_id] = 1 if pid in fleet_path_ids else 0
        _user_paths_mask[path_id] = 1 if pid in user_path_ids else 0
        _tp_index[path_id] = item.trip_id
        path_id += 1
        _tp_indptr[path_id] = path_id
        for eid in item.links:
            _lp_index[nnz] = eid
            nnz += 1
        _lp_indptr[path_id] = nnz
    for item in user_paths.path_iter():
        pid = item.path_id
        if pid in usable_path_ids:
            usable_path_ids.remove(pid)
            _user_paths_mask[path_id] = 1
            _tp_index[path_id] = item.trip_id
            path_id += 1
            _tp_indptr[path_id] = path_id
            for eid in item.links:
                _lp_index[nnz] = eid
                nnz += 1
            _lp_indptr[path_id] = nnz
    # if using all paths, there remaining paths are not usable:
    for item in all_paths.path_iter():
        pid = item.path_id
        if pid in usable_path_ids:
            _fleet_paths_mask[path_id] = 0
            _user_paths_mask[path_id] = 0
            _tp_index[path_id] = item.trip_id
            path_id += 1
            _tp_indptr[path_id] = path_id
            for eid in item.links:
                _lp_index[nnz] = eid
                nnz += 1
            _lp_indptr[path_id] = nnz
    printf("(%g seconds)\n", now() - t0)
    printf("Building sparse matrices...")
    t0 = now()
    lp_data = np.ones(nnz, dtype=np.uint8)
    tp_data = np.ones(n_recovered_paths, dtype=np.uint8)
    link_path = csc_matrix((lp_data, lp_index, lp_indptr), shape=(n_links, n_recovered_paths))
    trip_path = csc_matrix((tp_data, tp_index, tp_indptr), shape=(n_trips, n_recovered_paths))
    printf("(%g seconds)\n", now() - t0)
    # clean up
    igraph_vector_destroy(&path)
    printf("Returning sparse matrices and path masks\n")
    return link_path, trip_path, fleet_paths_mask, user_paths_mask


def get_epsilon_paths_all(PathDB paths,
                      Vector link_cost, Vector link_marginal_cost,
                      Vector trip_cost, Vector trip_marginal_cost, 
                      double epsilon_user, double epsilon_fleet,
                      bint use_all_paths=False,
                      unsigned int max_paths=0,
                      bint cfs_so=True):
    """Returns the set of paths within epsilon of best_trip_cost"""
    cdef Cursor cursor = paths.cursor()
    cdef unsigned int *p
    cdef unsigned int *bad_val
    cdef long int *val
    cdef igraph_vector_t path
    igraph_vector_init(&path, 0)
    cdef trip_id_storage_t trip
    cdef link_id_storage_t eid
    cdef char[:] buf

    cdef size_t i, n, n_links = vector_len(link_cost.vec), n_trips = vector_len(trip_cost.vec)
    cdef double cost, marginal_cost
    user_trip_counts = np.zeros(n_trips, dtype=np.uint)
    cdef unsigned long[:] _user_trip_counts = user_trip_counts
    fleet_trip_counts = np.zeros(n_trips, dtype=np.uint)
    cdef unsigned long[:] _fleet_trip_counts = fleet_trip_counts
    cdef size_t n_paths = 0, n_recovered_paths = 0, nnz = 0, n_fleet_paths = 0, n_user_paths = 0
    cdef uint8_t is_user_path, is_fleet_path
    print(f"Counting paths from {paths.name}...")
    printf("resetting cursor...\n")
    cursor.reset()
    cdef double t0 = now()
    cdef size_t empty_paths = 0
    while cursor.is_valid():
        cursor.populate()
        n_paths += 1
        is_fleet_path = 0
        is_user_path = 0
        trip = read_path_from_cursor(cursor, &path)
        n = vector_len(&path)
        if n == 0:
            empty_paths += 1
        else:
            cost = 0.0
            marginal_cost = 0.0
            for i in range(n):
                eid = <link_id_storage_t> vector_get(&path, i)
                cost += vector_get(link_cost.vec, eid)
                marginal_cost += vector_get(link_marginal_cost.vec, eid)
            if cost <= minimum_cost_threshold(vector_get(trip_cost.vec, trip), epsilon_user):
                is_user_path = 1
                n_user_paths += 1
                _user_trip_counts[trip] += 1
                if not cfs_so:
                    is_fleet_path = 1
                    n_fleet_paths += 1
                    _fleet_trip_counts[trip] += 1
            if cfs_so and (marginal_cost <= minimum_cost_threshold(vector_get(trip_marginal_cost.vec, trip), epsilon_fleet)):
                is_fleet_path = 1
                n_fleet_paths += 1
                _fleet_trip_counts[trip] += 1
            if use_all_paths or is_user_path or is_fleet_path:
                nnz += n
                n_recovered_paths += 1
        cursor.next()
    printf("Found %lu paths (%lu empty) in %g seconds; %lu usable (%lu fleet | %lu user).\n",
           n_paths, empty_paths, now()-t0, n_recovered_paths, n_fleet_paths, n_user_paths)
    cdef unsigned long _max_user_path_count = user_trip_counts.max()
    cdef unsigned long _min_user_path_count = user_trip_counts.min()
    printf("Trip counts (user recovered paths): min path count = %lu; max path count = %lu\n", _min_user_path_count, _max_user_path_count)
    cdef unsigned long _max_fleet_path_count = fleet_trip_counts.max()
    cdef unsigned long _min_fleet_path_count = fleet_trip_counts.min()
    printf("Trip counts (fleet recovered paths): min path count = %lu; max path count = %lu\n", _min_fleet_path_count, _max_fleet_path_count)
    assert _min_fleet_path_count > 0 and _min_user_path_count > 0 , "No paths found for some trips! Try a larger epsilon."
    lp_index = np.zeros(nnz, dtype=np.int64)
    lp_indptr = np.zeros(n_recovered_paths + 1, dtype=np.int64)
    cdef int64_t[:] _lp_index = lp_index
    cdef int64_t[:] _lp_indptr = lp_indptr
    tp_index = np.zeros(n_recovered_paths, dtype=np.int32)
    tp_indptr = np.zeros(n_recovered_paths+ 1, dtype=np.int32)
    cdef int32_t[:] _tp_index = tp_index
    cdef int32_t[:] _tp_indptr = tp_indptr

    fleet_paths = np.zeros(n_recovered_paths, dtype=np.bool)
    user_paths = np.zeros(n_recovered_paths, dtype=np.bool)
    cdef uint8_t[:] _fleet_paths = fleet_paths
    cdef uint8_t[:] _user_paths = user_paths

    printf("Populating sparse matrices...")
    cursor.reset()
    n_paths = 0
    n_recovered_paths = 0
    nnz = 0
    t0 = now()
    while cursor.is_valid():
        cursor.populate()
        n_paths += 1
        is_fleet_path = 0
        is_user_path = 0
        trip = read_path_from_cursor(cursor, &path)
        n = vector_len(&path)
        if n == 0:
            continue
        cost = 0.0
        marginal_cost = 0.0
        for i in range(n):
            eid = <link_id_storage_t> vector_get(&path, i)
            cost += vector_get(link_cost.vec, eid)
            marginal_cost += vector_get(link_marginal_cost.vec, eid)
        if cost <= minimum_cost_threshold(vector_get(trip_cost.vec, trip), epsilon_user):
            is_user_path = 1
            if not cfs_so:
                is_fleet_path = 1
        if cfs_so and (marginal_cost <= minimum_cost_threshold(vector_get(trip_marginal_cost.vec, trip), epsilon_fleet)):
            is_fleet_path = 1
        if use_all_paths or is_user_path or is_fleet_path:
            _fleet_paths[n_recovered_paths] = is_fleet_path
            _user_paths[n_recovered_paths] = is_user_path
            tp_index[n_recovered_paths] = trip
            n_recovered_paths += 1
            tp_indptr[n_recovered_paths] = n_recovered_paths
            for i in range(n):
                eid = <link_id_storage_t> vector_get(&path, i)
                lp_index[nnz] = eid
                nnz += 1
            lp_indptr[n_recovered_paths] = nnz
        cursor.next()
    printf("(%g seconds)\n", now() - t0)
    printf("Building sparse matrices...")
    t0 = now()
    lp_data = np.ones(nnz, dtype=np.uint8)
    tp_data = np.ones(n_recovered_paths, dtype=np.uint8)
    link_path = csc_matrix((lp_data, lp_index, lp_indptr), shape=(n_links, n_recovered_paths))
    trip_path = csc_matrix((tp_data, tp_index, tp_indptr), shape=(n_trips, n_recovered_paths))
    printf("(%g seconds)\n", now() - t0)
    printf("Returning sparse matrices and path masks\n")
    return link_path, trip_path, fleet_paths, user_paths


def get_projected_link_flow(Vector link_flow, Vector trip_volume,
                            LinkCost link_cost_fn,
                            trip_path, link_path, so_paths):
    n_trips, n_paths = trip_path.shape
    n_links, _ = link_path.shape
    printf("Creating variables...\n")
    f = cp.Variable(n_paths, nonneg=True)
    projected_flow = link_path @ f
    flow = np.clip(link_flow.to_array(), 0, None)
    q = trip_volume.to_array()
    total_volume = q.sum()
    max_link_flow = flow.max()
    error = (projected_flow - flow)
    try:
        weights = np.divide(
            link_cost_fn.free_flow_travel_time.to_array(),
            link_cost_fn.capacity.to_array()
        )
    except AttributeError:
        weights = 1.0
    constraints = [
        trip_path @ f == q
    ]
    if not so_paths.all():
        constraints.append(f[~so_paths] == 0.0)
    unused_links = flow == 0.0
    #if unused_links.any():
    #    constraints.append(projected_flow[unused_links] == 0.0)
    printf("Creating cvxpy problem...\n")
    objective = cp.sum_squares(cp.multiply(weights, error))
    print(f"is objective dcp? {objective.is_dcp()}")
    problem = cp.Problem(
        cp.Minimize(objective),
        constraints
    )
    printf("SOLVING FOR PROJECTED LINK FLOW\n")
    problem.solve(solver=cp.GUROBI, verbose=False)
    print("RESULTS:")
    total_flow_error = cp.sum(cp.abs(error)).value
    percent_error = 100 * cp.abs(error / (flow + unused_links)).value
    i = np.argmax(percent_error)
    j = np.argmax(cp.abs(error).value)
    print(f"Total misplaced flow: {total_flow_error} ({total_flow_error/n_links} per link) ({100 * total_flow_error / total_volume}% of total volume)")
    print(f"Projected link flow has max abs percent error: {percent_error[i]}% (|{error.value[i]}|/{flow[i]})")
    print(f"    max abs error: {error.value[j]} ({100 * error.value[j] / flow[j]}%)")
    return projected_flow.value, f.value, cp.max(cp.abs(error)).value

def critical_fleet_size_mip(link_flow, trip_volume, epsilon,
                        link_path, trip_path,
                        so_paths, user_paths,
                        link_cost, link_cost_gradient, marginal_link_cost,
                        link_flow_epsilon,
                        projected_link_flow,
                        beta=1.0,
                        ub=None,
                        lb=None,
                        cfs_so=True
                        ):
    n_trips, n_paths = trip_path.shape
    n_links, _ = link_path.shape

    # max or min?
    sense = cp.Minimize if cfs_so else cp.Maximize
    if beta < 0:
        raise Exception(f"Received negative value of beta: {beta}")
    if not cfs_so:
        # we are maximizing, so the penalty term should be negative
        beta = -beta
    print(f"beta={beta}")

    # variables
    f_user = cp.Variable(n_paths, nonneg=True, name="user_path_flow")
    f_fleet = cp.Variable(n_paths, nonneg=True, name="fleet_path_flow")
    fleet_paths = cp.Variable(n_paths, boolean=True, name="fleet_paths")
    # fleet_paths = so_paths.copy()
    least_fleet_marginal_cost = cp.Variable(n_trips, nonneg=True, name="fleet_marginal_cost")

    # constants
    m1 = 10.0 * np.max(link_path.T @ marginal_link_cost)
    m2 = np.max(trip_volume)

    # definitional constraints
    user_trip_flow = trip_path @ f_user
    user_link_flow = link_path @ f_user
    fleet_link_flow = link_path @ f_fleet
    user_impact = cp.multiply(user_link_flow, link_cost_gradient)
    fleet_impact = cp.multiply(fleet_link_flow, link_cost_gradient)

    if cfs_so:
        fleet_marginal_path_cost = link_path.T @ (marginal_link_cost - user_impact)
    else:
        fleet_marginal_path_cost = link_path.T @ (link_cost + fleet_impact)
    fleet_marginal_path_cost_bound = trip_path.T @ least_fleet_marginal_cost

    fleet_trip_flow = trip_path @ f_fleet
    fleet_link_flow = link_path @ f_fleet

    #link_flow_error = user_link_flow + fleet_link_flow - link_flow
    aggregate_link_flow = user_link_flow + fleet_link_flow
    #link_flow_lower_bound = np.minimum(link_flow, projected_link_flow)
    #link_flow_upper_bound = np.maximum(link_flow, projected_link_flow)
    total_fleet_volume = cp.sum(f_fleet)
    total_volume = trip_volume.sum()
    objective = sense(
        (total_fleet_volume + beta * cp.norm2(aggregate_link_flow - link_flow))
        / total_volume
    )
    constraints = []
    if not so_paths.all():
        constraints.append(f_fleet[~so_paths] == 0.0)
    constraints.extend([
        f_user <= m2 * user_paths,
        f_fleet <= m2 * fleet_paths,
        #fleet_marginal_path_cost >= cp.multiply(fleet_marginal_path_cost_bound, 1 + epsilon * (1 - fleet_paths)),
        fleet_marginal_path_cost >= fleet_marginal_path_cost_bound,
        fleet_marginal_path_cost <= fleet_marginal_path_cost_bound*(1+epsilon) + m1 * (1 - fleet_paths),
        user_trip_flow + fleet_trip_flow == trip_volume,
        #link_flow_error <= link_flow_epsilon,  # link flow is close
        #-link_flow_epsilon <= link_flow_error,
        #link_flow_lower_bound <= aggregate_link_flow,
        #aggregate_link_flow <= link_flow_upper_bound,
    ])
    if lb is not None:
        print(f"Adding objective lower bound obj >= {lb}")
        constraints.append(total_fleet_volume >= lb - 1e-6)
    if ub is not None:
        print(f"Adding objective upper bound obj <= {ub}")
        constraints.append(total_fleet_volume <= ub + 1e-6)
    return CriticalFleetSizeProblem(
        cp.Problem(
            objective,
            constraints
        ),
        f_fleet,
        f_user,
        fleet_link_flow,
        user_link_flow,
        link_flow,
        fleet_marginal_path_cost,
        fleet_paths
    )


def critical_fleet_size_lp(link_flow, trip_volume, epsilon,
                        link_path, trip_path,
                        fleet_paths, user_paths,
                        link_cost, link_cost_gradient, marginal_link_cost,
                        link_flow_epsilon,
                        projected_link_flow,
                        min_control_ratio=False,
                        beta=100.0,
                        grad_scale=1,
                        flow_scale=1,
                        grad_cutoff=0.0,
                        link_error_as_constraint=False,
                        cfs_so=True,
                        trip_cost=None,
                        ):
    n_trips, n_paths = trip_path.shape
    n_links, _ = link_path.shape

    # max or min?
    sense = cp.Minimize if cfs_so else cp.Maximize
    if not cfs_so:
        # we are maximizing, so the penalty term should be negative
        beta = -beta
    print(f"beta={beta}")

    # variables
    f_user = cp.Variable(n_paths, nonneg=True, name="user_path_flow")
    f_fleet = cp.Variable(n_paths, nonneg=True, name="fleet_path_flow")
    least_fleet_marginal_cost = cp.Variable(n_trips, nonneg=True, name="fleet_marginal_cost")
    if trip_cost is not None:
        print("setting min trip fleet marginal cost to trip cost")
        least_fleet_marginal_cost.project_and_assign(trip_cost)
    #mixture = cp.Variable(nonneg=True, name="mixture")

    # definitional constraints
    user_trip_flow = trip_path @ f_user
    user_link_flow = link_path @ f_user
    fleet_link_flow = link_path @ f_fleet
    link_cost_gradient = abs_relu(grad_cutoff, link_cost_gradient)
    user_impact = cp.multiply(user_link_flow, link_cost_gradient)
    fleet_impact = cp.multiply(fleet_link_flow, link_cost_gradient)
    print(f"link cost gradient range: [{link_cost_gradient[link_cost_gradient>0].min()}, {link_cost_gradient.max()}]")
    print(f"trip volume range: [{trip_volume.min()}, {trip_volume.max()}]")
    print(f"link flow range (non-zero): [{link_flow[link_flow>0].min()}, {link_flow.max()}] ")

    if cfs_so:
        fleet_marginal_path_cost = link_path.T @ (marginal_link_cost - user_impact)
    else:
        fleet_marginal_path_cost = link_path.T @ (link_cost + fleet_impact)
    fleet_marginal_path_cost_bound = trip_path.T @ least_fleet_marginal_cost

    fleet_trip_flow = trip_path @ f_fleet
    fleet_link_flow = link_path @ f_fleet

    aggregate_link_flow = user_link_flow + fleet_link_flow
    #link_flow_lower_bound = np.minimum(link_flow, projected_link_flow)
    #link_flow_upper_bound = np.maximum(link_flow, projected_link_flow)
    #feasible_link_flow = mixture * projected_link_flow + (1.0 - mixture) * link_flow

    constraints = []
    if not user_paths.all():
        constraints.append(f_user[~user_paths] == 0.0)
    if not fleet_paths.all():
        constraints.append(f_fleet[~fleet_paths] == 0.0)
    constraints.extend([
        user_trip_flow + fleet_trip_flow == trip_volume,
        #link_flow_error <= link_flow_epsilon,  # link flow is close
        #-link_flow_epsilon <= link_flow_error,
        #user_link_flow + fleet_link_flow == feasible_link_flow,
        #mixture <= 1.0,
        #link_flow_lower_bound <= aggregate_link_flow,
        #aggregate_link_flow <= link_flow_upper_bound,
    ])
    if not min_control_ratio:
        constraints.extend([
            fleet_marginal_path_cost >= fleet_marginal_path_cost_bound,
            fleet_marginal_path_cost[fleet_paths] <= fleet_marginal_path_cost_bound[fleet_paths] * (1 + epsilon),
        ])
    print("Building cvxpy problem")
    print(f"trip volume = {trip_volume.sum()}")
    fleet_fraction = cp.sum(f_fleet) #/ trip_volume.sum()
    link_flow_error = aggregate_link_flow - link_flow #/ trip_volume.sum()
    total_volume = trip_volume.sum()
    if not link_error_as_constraint:
        penalty = cp.norm2(link_flow_error)
    else:
        link_flow_abs_error = abs(link_flow - projected_link_flow).max()
        link_error_bound = cp.Variable(n_links, nonneg=True, name="link_error_bound")
        link_error_bound.project_and_assign(link_flow_abs_error)
        print("Adding link flow error box constraints")
        constraints.extend([
            link_flow_error <= link_error_bound,
            link_flow_error >= -link_error_bound,
        ])
        penalty = cp.sum(link_error_bound)
    if beta < np.inf:
        objective = sense(
            (fleet_fraction + beta * penalty) / total_volume
        )
    else:
        objective = sense(fleet_fraction / total_volume)
        constraints.append(penalty == 0.0)
    return CriticalFleetSizeProblem(
        cp.Problem(
            objective,
            constraints,
        ),
        f_fleet,
        f_user,
        fleet_link_flow,
        user_link_flow,
        link_flow,
        fleet_marginal_path_cost,
    )

def abs_relu(threshold, x):
    return x * (abs(x) > np.clip(threshold, 0, None))
