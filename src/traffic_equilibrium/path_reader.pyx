# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True,
from .solver cimport Problem
from libc.stdio cimport fopen, fclose, fread, FILE, printf
from .zlib cimport gzopen, gzFile, gzclose, gzfread
from .dang cimport dang_index_t
from .pathdb cimport PathDB

from cpython cimport array
import array

import time

def read_paths_from_files(Problem problem, list fnames, str db_fname):
    cdef str fname
    cdef long int number_of_links = problem.network.number_of_links()
    cdef long int number_of_trips = problem.demand.number_of_trips()
    cdef PathDB db = PathDB(db_fname)
    for fname in fnames:
        read_paths(number_of_links, number_of_trips, fname, db)
    db.close()


def read_paths(long int number_of_links, long int number_of_trips, str fname,
               PathDB db):
    cdef gzFile stream = gzopen(fname.encode("ascii"), 'r')
    cdef dang_index_t element, delimiter = -1
    cdef long int n_paths = 0, commit_every = int(25e6)
    cdef bint next_read_is_trip_id = True
    cdef bint is_new
    cdef array.array path = array.array('I')
    cdef array.array trip_id = array.array('I')
    cdef double t1, t0 = time.time()
    while gzfread(&element, sizeof(element), 1, stream):
        if element == delimiter:
            next_read_is_trip_id = True
            # add path to db
            db.put(path.tobytes(), trip_id.tobytes())
            # clear path array
            del path[:]
            # clear trip id array
            trip_id.pop()
            n_paths += 1
            # commit to the db
            if n_paths % commit_every == 0:
                printf("Committing %li paths to the database.\n", commit_every)
                t1 = time.time()
                db.commit()
                t1 = time.time() - t1
                printf("Committed %li paths to the database in %g seconds.\n",
                       commit_every, t1)
        else:
            if next_read_is_trip_id:
                next_read_is_trip_id = False
                trip_id.append(element)
            else:
                path.append(element)
    printf("Committing final batch.")
    db.commit()
    t1 = time.time() - t0
    gzclose(stream)
    printf("Added all %li paths to the path database in %g seconds.\n",
           n_paths, t1)

