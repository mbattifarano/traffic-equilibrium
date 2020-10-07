# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

DEF NSEC_PER_SEC = 1000000000.0

cdef inline double now() nogil:
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec + (ts.tv_nsec / NSEC_PER_SEC)
