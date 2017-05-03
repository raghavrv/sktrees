# cython: wraparound=False
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=False

cdef class MSE:
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, Node* nodes, double* node_stats,
                  double weighted_n_samples):
        self.nodes = nodes
        self.node_stats = node_stats
        self.split_records = split_records
        self.weighted_n_samples = weighted_n_samples
        self.n_stats = 3

    cdef double _impurity(self, SIZE_t node_id) nogil:  # except ?
        # If the formula for variance is expanded, it boils down to
        #       sum(yi ^ 2)      _                _
        #      -------------  -  y ^ 2,     where y is the mean of {yi}
        #          sum(wi)
        return node_stats[2] / node.weighted_n_node_samples - node_stats[0] ** 2.0

    cdef int _store_children_impurities(self, SIZE_t parent_node,
                                        SIZE_t record_index) nogil except -1:
        cdef:
            SplitRecord split = self.split_records[record_index]

        split.impurity_right = ((split.sum_ysq_right / split.weighted_n_right) -
                                (split.sum_y_right / split.weighted_n_right) ** 2)
        split.impurity_left = ((split.sum_ysq_left / split.weighted_n_left) -
                               (split.sum_y_left / split.weighted_n_left) ** 2)
        return 0

    # A faster proxy to actual impurity improvement to speed up computations
    cdef double _proxy_improvement(self, SIZE_t parent_node, SIZE_t record_index) nogil:
        cdef:
            SplitRecord split = self.split_records[record_index]

        self._store_children_impurities(parent_node, record_index)
        return (-split.weighted_n_right * split.impurity_right -
                split.weighted_n_left * split.impurity_left)

    # Actual mse impurity improvement without proxies
    cdef double _impurity_improvement(self, SIZE_t node_id, SIZE_t record_index) nogil:
        cdef:
            double improvement
            Node node = self.nodes[node_id]

        # Parent impurity
        improvement = node.impurity
        # Weighted average of the right / left impurity
        improvement -= self._proxy_improvement(node_id, record_index) / node.weighted_n_node_samples
        # Weigh this quantity by the node weight fraction
        return improvement * node.weighted_n_node_samples / self.weighted_n_samples
