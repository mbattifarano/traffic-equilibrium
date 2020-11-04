# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,

from libc.stdlib cimport free, malloc
from libc.stdio cimport stdout, setbuf, printf, fopen, fclose
from .igraph cimport *
from .vector cimport Vector
from .igraph_utils cimport *

import os
import json

# disable stdout buffer (for debugging)
# setbuf(stdout, NULL)


cdef class GraphInfo:
    cdef readonly str name
    cdef readonly graph_index_t number_of_nodes
    cdef readonly graph_index_t number_of_links

    def __cinit__(self, str name, graph_index_t number_of_nodes, graph_index_t number_of_links):
        self.name = name
        self.number_of_nodes = number_of_nodes
        self.number_of_links = number_of_links


    def __str__(self):
        return f"GraphInfo(name={self.name}, number_of_nodes={self.number_of_nodes}, number_of_links={self.number_of_links})"


cdef class DiGraph:
    def __cinit__(self, str name=""):
        cdef igraph_integer_t number_of_nodes = 0
        cdef igraph_bool_t is_directed = 1
        self.name = name
        self.graph = <igraph_t*> malloc(sizeof(igraph_t))
        igraph_empty(
            self.graph,
            number_of_nodes,
            is_directed
        )

    def __dealloc__(self):
        if self.graph is not NULL:
            igraph_destroy(self.graph)
            free(self.graph)
            self.graph = NULL

    cdef str get_name(self):
        return self.name

    cdef void set_name(self, str name):
        self.name = name

    cdef graph_index_t number_of_nodes(self) nogil:
        return igraph_vcount(self.graph)

    cdef graph_index_t number_of_links(self) nogil:
        return igraph_ecount(self.graph)

    cdef void add_nodes(self, igraph_integer_t number_of_nodes) nogil:
        igraph_add_vertices(self.graph, number_of_nodes, NULL)

    cdef void add_links(self, igraph_vector_t *links) nogil:
        igraph_add_edges(self.graph, links, NULL)

    def append_nodes(self, igraph_integer_t n):
        return self.add_nodes(n)

    def add_links_from(self, links):
        cdef long int i, n_links = len(links)
        cdef igraph_vector_t vec
        igraph_vector_init(&vec, 2 * n_links)
        for i in range(n_links):
            u, v = links[i]
            vector_set(&vec, 2 * i, u)
            vector_set(&vec, 2 * i + 1, v)
        self.add_links(&vec)
        igraph_vector_destroy(&vec)

    def degree_of(self, igraph_integer_t vid):
        cdef igraph_vs_t vs = igraph_vss_1(vid)
        cdef igraph_vector_t result
        cdef int deg
        igraph_vector_init(&result, 1)
        igraph_degree(self.graph, &result, vs, igraph_neimode_t.IGRAPH_OUT, 0)
        deg = <int> vector_get(&result, 0)
        igraph_vector_destroy(&result)
        return deg

    def links(self):
        l = []
        cdef graph_index_t i
        cdef igraph_integer_t u, v
        for i in range(self.number_of_links()):
            igraph_edge(self.graph, i, &u, &v)
            l.append((int(u), int(v)))
        return l

    def info(self):
        return GraphInfo(
            self.name,
            self.number_of_nodes(),
            self.number_of_links()
        )

    def save(self, dirname):
        cdef list edgelist = []
        cdef long int eid, n = self.number_of_links()
        cdef int u, v
        for eid in range(n):
            igraph_edge(self.graph, eid, &u, &v)
            edgelist.append((int(u), int(v)))
        with open(os.path.join(dirname, f"network.json"), "w") as fp:
            json.dump({
                'name': self.name,
                'number_of_nodes': self.number_of_nodes(),
                'edges': edgelist,
            }, fp, indent=2)

    @staticmethod
    def load(dirname):
        with open(os.path.join(dirname, f"network.json")) as fp:
            data = json.load(fp)
        cdef DiGraph network = DiGraph.__new__(DiGraph, data["name"])
        network.add_nodes(data["number_of_nodes"])
        network.add_links_from(data["edges"])
        return network
