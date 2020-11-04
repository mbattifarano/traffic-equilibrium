from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand
from .path_set cimport PathSet

cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PathSet paths,
                                    Vector best_path_cost
                                    ) nogil

cdef PointerVector init_path_vectors(OrgnDestDemand demand)
