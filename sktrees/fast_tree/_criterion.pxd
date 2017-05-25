# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
np.import_array()

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_float32 DTYPE_t          # Type of X


from sklearn.tree._tree cimport Node
from sktrees.fast_tree._data_structures cimport SplitRecord
from sktrees.fast_tree._data_structures cimport SplitRecordStore
from sktrees.fast_tree._data_structures cimport StoppingCriteria
from sktrees.fast_tree._data_structures cimport SplitRecord_SIZE_BYTES


cdef SIZE_t INVALID = -1


cdef class MSE:
    cdef SIZE_t n_outputs

    cdef SIZE_t node_stride
    cdef SIZE_t split_stride

    cdef double* node_stats  # The node stats will have the sum_y, sum_ysq, mean_y informations
    cdef SplitRecord* split_records  # Will be used to store the right/left partition's stats
    cdef double weighted_n_samples
    cdef SIZE_t n_stats      # The number of stats : 3 (mean_y, sum_y, sum_ysq)
    cdef SIZE_t* expanding_nodes

    cdef SIZE_t* samples_to_node_idx_map

    cdef int store_node_impurities(self, SIZE_t start_node_idx,
                                   SIZE_t end_node_idx) nogil except -1
    cdef int store_children_impurities(self, SIZE_t start_node_idx,
                                       SIZE_t end_node_idx) nogil except -1
    # Proxy to speed up computation
    cdef int store_proxy_impurity_improvements(
        self, SIZE_t start_node_idx, SIZE_t end_node_idx) nogil except -1
    cdef int store_one_node_proxy_impurity_improvement(
                self, SIZE_t node_idx, SplitRecord* this_record,
                double* stats_left, double* stats_right) nogil except -1
    cdef int find_best_split(self, DTYPE_t[:, :] X,
                             DOUBLE_t[:, :] y,
                             double[:] sample_weight,
                             SIZE_t[:, :] X_idx_sorted,
                             SIZE_t[:] shuffled_features,
                             SIZE_t n_features,
                             StoppingCriteria stopping) nogil except -1
