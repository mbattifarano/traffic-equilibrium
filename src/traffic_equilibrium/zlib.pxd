cdef extern from "zlib.h" nogil:
    cdef struct gzFile_s:
        pass

    ctypedef gzFile_s gzFile

    cdef gzFile gzopen(const char *path, const char *mode)
    cdef int gzclose(gzFile file)
    cdef int gzfread(void *buf, size_t size, size_t nitems, gzFile file)