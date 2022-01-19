from libc.stdint cimport uint8_t

cdef extern from "leveldb/c.h" nogil:
    # types
    ctypedef struct leveldb_t:
        pass

    ctypedef struct leveldb_options_t:
        pass

    ctypedef struct leveldb_writeoptions_t:
        pass

    ctypedef struct leveldb_readoptions_t:
        pass

    ctypedef struct leveldb_writebatch_t:
        pass

    ctypedef struct leveldb_iterator_t:
        pass

    # database functions
    leveldb_t* leveldb_open(const leveldb_options_t* options,
                            const char* name, char** errptr)
    void leveldb_destroy_db(const leveldb_options_t* options,
                            const char* name, char** errptr)
    
    void leveldb_close(leveldb_t* db)

    void leveldb_put(leveldb_t* db,
                     const leveldb_writeoptions_t* options,
                     const char* key, size_t keylen,
                     const char* val, size_t vallen,
                     char** errptr)
    
    void leveldb_write(leveldb_t* db,
                       const leveldb_writeoptions_t* options,
                       leveldb_writebatch_t* batch,
                       char** errptr)
    
    
    # Returns NULL if not found.  A malloc()ed array otherwise.
    # Stores the length of the array in *vallen.
    char* leveldb_get(leveldb_t* db,
                      const leveldb_readoptions_t* options,
                      const char* key, size_t keylen, size_t* vallen,
                      char** errptr)

    # iterators
    leveldb_iterator_t* leveldb_create_iterator(leveldb_t* db,
                                                const leveldb_readoptions_t* options)
    void leveldb_iter_destroy(leveldb_iterator_t*)
    void leveldb_iter_seek_to_first(leveldb_iterator_t*)
    void leveldb_iter_next(leveldb_iterator_t*)
    const char* leveldb_iter_key(const leveldb_iterator_t*, size_t* klen)
    const char* leveldb_iter_value(const leveldb_iterator_t*, size_t* vlen)
    bint leveldb_iter_valid(const leveldb_iterator_t*)

    # options
    leveldb_options_t* leveldb_options_create()
    void leveldb_options_destroy(leveldb_options_t*)
    void leveldb_options_set_create_if_missing(leveldb_options_t*, uint8_t)
    void leveldb_options_set_write_buffer_size(leveldb_options_t*, size_t)
    void leveldb_options_set_max_file_size(leveldb_options_t*, size_t)
    void leveldb_options_set_block_size(leveldb_options_t*, size_t)
    
    # write options
    leveldb_writeoptions_t* leveldb_writeoptions_create()
    void leveldb_writeoptions_destroy(leveldb_writeoptions_t*)
    void leveldb_writeoptions_set_sync(leveldb_writeoptions_t*, uint8_t)
    
    # read options
    leveldb_readoptions_t* leveldb_readoptions_create()
    void leveldb_readoptions_destroy(leveldb_readoptions_t*)
    void leveldb_readoptions_set_verify_checksums(leveldb_readoptions_t*, uint8_t)
    void leveldb_readoptions_set_fill_cache(leveldb_readoptions_t*, uint8_t)

    # write batch
    leveldb_writebatch_t* leveldb_writebatch_create()
    
    void leveldb_writebatch_destroy(leveldb_writebatch_t* batch)
    
    void leveldb_writebatch_clear(leveldb_writebatch_t* batch)

    void leveldb_writebatch_put(leveldb_writebatch_t* batch,
                                const char* key, size_t klen,
                                const char* val, size_t vlen)
