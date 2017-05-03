# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause

cdef class MultiSplitter:
    def __cinit__(self, Node* nodes, double* node_stats,
                  double weighted_n_samples):
        self.nodes = nodes
        self.split_records = NULL
        self.shuffled_features = NULL

    def __dealloc__(self):
        free(self.split_records)

    def 




cdef int init_SplitRecord_from_parent(SIZE_t record_idx,
                                      SIZE_t feature,
                                      SIZE_t start, SIZE_t end,
                                      SIZE_t parent_id,
                                      bint all_at_right=1) nogil except -1:
    cdef:
        SplitRecord s = self.split_records[record_idx]

    s.feature = feature
    s.start = start
    s.end = end
    s.pos = 0
    s.threshold = np.nan
    s.proxy_improvement = -1
    s.improvement = -1

    s.best_feature = 0
    s.best_pos = 0
    s.best_threshold = np.nan
    s.best_proxy_improvement = -1
    s.best_improvement = -1

    # All samples are on the right, if all_at_right else on the left
    if all_at_right:
        s.n_left = 0
        s.weighted_n_left = 0
        s.sum_y_left = 0
        s.sum_ysq_left = 0
        s.impurity_left = 0

        s.n_right = parent_node.n_node_samples
        s.weighted_n_right = parent_node.weighted_n_node_samples
        s.sum_y_right = parent_node_stats[1]
        s.sum_ysq_right = parent_node_stats[2]
        s.impurity_right = parent_node.impurity
    else:
        s.n_right = 0
        s.weighted_n_right = 0
        s.sum_y_right = 0
        s.sum_ysq_right = 0
        s.impurity_right = 0

        s.n_left = parent_node.n_node_samples
        s.weighted_n_left = parent_node.weighted_n_node_samples
        s.sum_y_left = parent_node_stats[1]
        s.sum_ysq_left = parent_node_stats[2]
        s.impurity_left = parent_node.impurity
    return 0