# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause


from cpython cimport Py_INCREF, PyObject
from libc.stdlib cimport malloc, free, calloc, realloc

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.externals import joblib

ctypedef np.npy_intp SIZE_t
ctypedef np.npy_float64 DOUBLE_t


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

cdef struct StoppingCriteria:
    SIZE_t min_samples_split
    SIZE_t min_samples_leaf
    SIZE_t min_weight_leaf
    double min_weight_fraction_leaf
    SIZE_t max_leaf_nodes
    double min_impurity_decrease
    # TODO Deprecated, should we support it?
    double min_impurity_split

cdef struct SplitRecord:
    # Data to track sample splitting process
    # This structure also store the best split found so far
    SIZE_t feature         # Which feature to split on.
    SIZE_t start
    SIZE_t end
    SIZE_t prev_pos
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double weighted_n_node_samples
    double node_impurity
    double threshold       # Threshold to split at.
    double proxy_improvement  # Proxy for impurity improvement to speed up
                              # computation times
    double improvement     # Impurity improvement given parent node.

    # We maintain a copy of the best so far
    SIZE_t best_feature
    SIZE_t best_pos
    double best_threshold
    double best_proxy_improvement
    # This will be updated only finally to save some computations
    double best_improvement

    # stats for left partition
    SIZE_t n_left
    double weighted_n_left
    double impurity_left
    double proxy_impurity_left

    # stats for right partition
    SIZE_t n_right
    double weighted_n_right
    double impurity_right
    double proxy_impurity_right


cdef SIZE_t SplitRecord_SIZE_BYTES = sizeof(SplitRecord) / 8


# We make a numpy struct dtype for converting the array of SplitRecord structs
# to a python numpy array
SPLITRECORD_NP_STRUCT_DTYPE = np.dtype({
    'names': ['feature', 'start', 'end', 'pos', 'threshold',
              'proxy_improvement',
              'best_feature', 'best_pos', 'best_threshold',
              'best_proxy_improvement', 'best_improvement',
              'n_left', 'weighted_n_left', 'n_right', 'weighted_n_right'],
    'formats': [np.intp, np.intp, np.intp, np.intp, np.float64,
                np.float64,
                np.intp, np.intp, np.float64,
                np.float64, np.float64,
                np.intp, np.float64, np.intp, np.float64],
    'offsets': [
        <Py_ssize_t> &((<SplitRecord*> NULL).feature),
        <Py_ssize_t> &((<SplitRecord*> NULL).start),
        <Py_ssize_t> &((<SplitRecord*> NULL).end),
        <Py_ssize_t> &((<SplitRecord*> NULL).pos),
        <Py_ssize_t> &((<SplitRecord*> NULL).threshold),
        <Py_ssize_t> &((<SplitRecord*> NULL).proxy_improvement),
        <Py_ssize_t> &((<SplitRecord*> NULL).best_feature),
        <Py_ssize_t> &((<SplitRecord*> NULL).best_pos),
        <Py_ssize_t> &((<SplitRecord*> NULL).best_threshold),
        <Py_ssize_t> &((<SplitRecord*> NULL).best_proxy_improvement),
        <Py_ssize_t> &((<SplitRecord*> NULL).best_improvement),
        <Py_ssize_t> &((<SplitRecord*> NULL).n_left),
        <Py_ssize_t> &((<SplitRecord*> NULL).weighted_n_left),
        <Py_ssize_t> &((<SplitRecord*> NULL).n_right),
        <Py_ssize_t> &((<SplitRecord*> NULL).weighted_n_right),
    ]
})


cdef class SplitRecordStore:
    """This class maintains an array of SplitRecords for the tree builder"""
    cdef SplitRecord* array
    cdef SIZE_t n_records
    cdef int _resize(self, size_t n_records) except -1
    cdef np.ndarray _get_split_record_ndarray(self)


ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (SplitRecord*)
    (SplitRecord**)
    (SIZE_t*)
    (DOUBLE_t*)


cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *
