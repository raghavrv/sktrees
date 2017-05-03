# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef struct SplitRecord:
    # Data to track sample splitting process
    # This structure also store the best split found so far
    SIZE_t feature         # Which feature to split on.
    SIZE_t start
    SIZE_t end
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double proxy_improvement  # Proxy for impurity improvement to speed up
                              # computation times
    double improvement     # Impurity improvement given parent node.

    # Use these to compare the current split stats with the best so far
    SIZE_t best_feature
    SIZE_t best_pos
    double best_threshold
    double best_proxy_improvement
    # This will be updated only finally to save some computations
    double best_improvement

    # stats for left partition
    SIZE_t n_left
    double weighted_n_left
    double sum_y_left      # sum of target values of left split
    double sum_ysq_left    # sum of square of target values of left

    # stats for right partition
    SIZE_t n_right
    double weighted_n_right
    double sum_y_right     # sum of target values of right split
    double sum_ysq_right   # sum of square of target values of right


cdef inline int init_SplitRecord_from_parent(SplitRecord* s,
                                             SIZE_t feature,
                                             SIZE_t start, SIZE_t end,
                                             Node* parent_node,
                                             double* parent_node_stats,
                                             bint all_at_right=?) nogil except -1