from .graph cimport DiGraph
from .vector cimport Vector, PointerVector
from .trips cimport OrgnDestDemand
from .pathdb cimport PathDB

cdef void shortest_paths_assignment(DiGraph network,
                                    Vector cost,
                                    OrgnDestDemand demand,
                                    Vector flow,
                                    PathDB paths,
                                    Vector trip_cost
                                    ) nogil
