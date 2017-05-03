# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

from sklearn.tree._tree cimport Node
from sklearn.fast_tree._utils cimport SplitRecord

cdef class MSE:
    cdef Node* nodes   # The tree nodes, which will have impurity / weighted_n_node_samples
    cdef double* node_stats  # The node stats will have the sum_y, sum_ysq, mean_y informations
    cdef SplitRecord* split_records  # Will be used to store the right/left partition's stats
    cdef double weighted_n_samples
    cdef SIZE_t n_stats      # The number of stats : 3 (mean_y, sum_y, sum_ysq)

    cdef double _impurity(self, SIZE_t node_id) nogil
    cdef int _store_children_impurities(self, SIZE_t parent_node,
                                        SIZE_t record_index) nogil except -1
    # A faster proxy to actual impurity improvement to speed up computations
    cdef double _proxy_improvement(self, SIZE_t parent_node, SIZE_t record_index) nogil
    # Actual mse impurity improvement without proxies
    cdef double _impurity_improvement(self, SIZE_t node_id, SIZE_t record_index) nogil