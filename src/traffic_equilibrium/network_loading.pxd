from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand

cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PointerVector paths,
                                    ) nogil

cdef PointerVector init_path_vectors(OrgnDestDemand demand)
