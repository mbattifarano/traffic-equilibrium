"""Wrapper around scipy sparse"""
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.data import _data_matrix
from scipy.sparse.sputils import check_shape


class SparseSave:
    def save(self, fname, compressed=True):
        data = dict(
            indices=self.indices,
            indptr=self.indptr,
            data=self.data,
            shape=self.shape,
            format=self.format.encode('ascii')
        )
        if compressed:
            np.savez_compressed(fname, **data)
        else:
            np.savez(fname, **data)


def load(fname):
    with np.load(fname) as data:
        fmt = data['format'].item().decode('ascii')
        cls = {
            'csc': SparseColumnMatrix,
            'csr': SparseRowMatrix
        }[fmt]
        return cls((data['data'], data['indices'], data['indptr']), data['shape'])


class SparseColumnMatrix(csc_matrix, SparseSave):
    """Subclass csc_matrix with a specialized interface.

    Does not perform any checks on the data
    dtype and copy are ignored.
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        data, indices, indptr = arg1
        self.indices = indices
        self.indptr = indptr
        self.data = data

        if shape is None:
            major_dim = len(indptr) -1
            minor_dim = self.indices.max() + 1
            shape = self._swap((major_dim, minor_dim))
        self._shape = check_shape(shape)


        # self.check_format(full_check=False)


class SparseRowMatrix(csr_matrix, SparseSave):
    """Subclass csr_matrix with a specialized interface.

    Does not perform any checks on the data.
    dtype and copy are ignored.
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        data, indices, indptr = arg1

        self.indices = indices
        self.indptr = indptr
        self.data = data

        if shape is None:
            major_dim = len(indptr) -1
            minor_dim = self.indices.max() + 1
            shape = self._swap((major_dim, minor_dim))
        self._shape = check_shape(shape)

