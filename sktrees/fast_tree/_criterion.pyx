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

    @classmethod
    def get_node_stats_size(cls, SIZE_t n_outputs, SIZE_t n_expanding_nodes):
        # Number of nodes * the number of elements per node
        return n_expanding_nodes * (3 * (1 + n_outputs))

    def __cinit__(self, SplitRecordStore split_record_store,
                  np.ndarray node_stats,
                  np.ndarray expanding_nodes, SIZE_t n_expanding_nodes,
                  double weighted_n_samples, SIZE_t n_outputs,
                  np.ndarray shuffled_features,
                  SIZE_t start_feat_idx, SIZE_t end_feat_idx,
                  np.ndarray samples_to_node_idx_map):
        """The start_feat_index and end_feat_index marks the dimensional limits
        in which this criterion will work on for the current job"""

        self.split_records = <SplitRecord*>split_record_store.array
        self.node_stats = <double*>node_stats.data

        self.expanding_nodes = <SIZE_t*>expanding_nodes.data
        self.n_expanding_nodes = n_expanding_nodes

        self.samples_to_node_idx_map = <SIZE_t*>samples_to_node_idx_map.data

        self.weighted_n_samples = weighted_n_samples
        self.n_outputs = n_outputs

        # The number of stats that MSE requires per node per output per split
        # We use this to move from stats array of the right split,
        # to the whole node and then to the left split
        self.split_stride = 1 + n_outputs
        # n_total number of stats per node for all outputs.
        # The `node_stats[node_id * node_stride]` would contain data as
        # [sum_ysq_left_all, sum_y_left_op0, sum_y_left_op1,
        #  sum_ysq_node_all, sum_y_node_op0, sum_y_node_op1,
        #  sum_ysq_right_all, sum_y_right_op0, sum_ysq_right_op1]
        self.node_stride = 3 * self.split_stride

        self.start_feat_idx = start_feat_idx
        self.end_feat_idx = end_feat_idx
        self.current_feat_idx = start_feat_idx


    cdef int store_node_impurities(self,
                                   SIZE_t start_node_idx, SIZE_t end_node_idx) nogil except -1:
        """Computes the node impurities for current feature/ split pos for given range of nodes

        The impurities for expanding_nodes[start_node_idx:end_node_idx] are computed and
        stored in split_records[start_node_idx:end_node_idx]

        NOTE: node index != node id

        (The impurity values are available at record.node_impurity)

        If the formula for variance is expanded, it boils down to
              sum(yi ^ 2)      _                _
             -------------  -  y ^ 2,     where y is the mean of {yi}
                 sum(wi)
        """
        cdef:
            SIZE_t i, j  # variables for iteration counts

            SIZE_t* expanding_nodes = self.expanding_nodes

            double* this_stats
            SplitRecord* this_record = self.split_records + start_node_idx

            double* impurity_ptr
            double* weighted_n_node_samples_ptr

            double sum_ysq_node


        # To make it less verbose
        # This is a hack which references the member of the struct
        # We advance to the next using the sizeof(SplitRecord) / 8
        # XXX Is this unsafe / hacky? I can't find any reference for this online ;/
        impurity_ptr = &(this_record[0].node_impurity)
        weighted_n_node_samples_ptr = &(this_record[0].weighted_n_node_samples)

        # To advance to the stats for the next node do
        # this_stats += self.node_stride
        this_stats = self.node_stats
        this_stats += start_node_idx * self.node_stride


        for i in range(start_node_idx, end_node_idx):
            # Get the center (node) sum_ysq statistic
            sum_ysq_node = this_stats[self.split_stride]
            impurity_ptr[0] = sum_ysq_node / weighted_n_node_samples_ptr[0]

            for j in range(self.n_outputs):
                impurity_ptr[0] -= (this_stats[self.split_stride + (1 + j)] /
                                    weighted_n_node_samples_ptr[0])**2.0
                impurity_ptr[0] /= self.n_outputs

            # Instead of doing -
            # this_record += 1
            # impurity_ptr = &this_record.node_impurity
            # We advace using the member pointers
            impurity_ptr += SplitRecord_SIZE_BYTES
            weighted_n_node_samples_ptr += SplitRecord_SIZE_BYTES

            this_stats += self.node_stride

        return 0

    cdef int store_children_impurities(self, SIZE_t start_node_idx,
                                       SIZE_t end_node_idx) nogil except -1:
        cdef:
            cdef SIZE_t i, j
            SplitRecord* this_record = self.split_records + start_node_idx

            double* stats_left = self.node_stats + self.node_stride * start_node_idx
            double* stats_right = stats_left + 2 * self.split_stride

            double* impurity_left_ptr = &this_record[0].impurity_left
            double* impurity_right_ptr = &this_record[0].impurity_right

            double* weighted_n_left_ptr = &this_record[0].weighted_n_left
            double* weighted_n_right_ptr = &this_record[0].weighted_n_right

            double weighted_n_left_inv, weighted_n_right_inv

        for i in range(start_node_idx, end_node_idx):
            weighted_n_left_inv = 1. / weighted_n_left_ptr[0]
            weighted_n_right_inv = 1. / weighted_n_right_ptr[0]

            # stats_left[0] = sum_sq_left and stats_right[0] = sum_sq_right
            impurity_left_ptr[0] = stats_left[0] * weighted_n_left_inv
            impurity_right_ptr[0] = stats_right[0] * weighted_n_right_inv

            for j in range(self.n_outputs):
                impurity_left_ptr[0] -= (stats_left[1 + j] * weighted_n_left_inv) ** 2
                impurity_right_ptr[0] -= (stats_right[1 + j] * weighted_n_right_inv) ** 2

            # Advance all pointers to point to the data for next node
            impurity_left_ptr += SplitRecord_SIZE_BYTES
            impurity_right_ptr += SplitRecord_SIZE_BYTES
            weighted_n_left_ptr += SplitRecord_SIZE_BYTES
            weighted_n_right_ptr += SplitRecord_SIZE_BYTES

            stats_left += self.node_stride
            stats_right += self.node_stride
        return 0

    cdef int store_proxy_impurity_improvements(
                 self, SIZE_t start_node_idx, SIZE_t end_node_idx) nogil except -1:
        """Computes the node's proxy impurities for current feature/ split pos
        for given range of nodes

        The proxy impurities for expanding_nodes[start_node_idx:end_node_idx]
        are computed and stored in split_records[start_node_idx:end_node_idx]

        (The proxy impurity values are available at record.proxy_impurities)

        """
        cdef:
            SIZE_t i, j  # variables for iteration counts

            SIZE_t* expanding_nodes = self.expanding_nodes

            double* this_stats
            SplitRecord* this_record

            double proxy_impurity_left
            double proxy_impurity_right
            double* proxy_improvement_ptr

            double* weighted_n_left_ptr
            double* weighted_n_right_ptr

            double sum_ysq_node
            double weighted_n_node_samples

            double* sum_y_left
            double* sum_y_right

        this_record = self.split_records + start_node_idx
        this_stats = self.node_stats + start_node_idx * self.node_stride

        proxy_improvement_ptr = &this_record[0].proxy_improvement

        # this_stats[0] is sum_ysq_left, sum_y_left for each output starts from
        # the next index in that array
        sum_y_left = this_stats + 1
        sum_y_right = sum_y_left + 2 * self.split_stride

        weighted_n_left_ptr = &this_record[0].weighted_n_left
        weighted_n_right_ptr = &this_record[0].weighted_n_right

        for i in range(start_node_idx, end_node_idx):
            proxy_impurity_left = proxy_impurity_right = 0.0
            for j in range(self.n_outputs):
                proxy_impurity_left += sum_y_left[j] * sum_y_left[j]
                proxy_impurity_right += sum_y_right[j] * sum_y_right[j]

            proxy_improvement_ptr[0] = (proxy_impurity_left / weighted_n_left_ptr[0] +
                                        proxy_impurity_right / weighted_n_right_ptr[0])

            # Advance all the pointers / indices to the next node's data/record
            sum_y_left += self.node_stride
            sum_y_right += self.node_stride

            this_record += 1

            weighted_n_left_ptr += SplitRecord_SIZE_BYTES
            weighted_n_right_ptr += SplitRecord_SIZE_BYTES

        return 0

    cdef int store_one_node_proxy_impurity_improvement(
                self, SIZE_t node_idx, SplitRecord* this_record,
                double* stats_left, double* stats_right) nogil except -1:
        # NOTE node_idx != node_id
        # expanding_nodes[node_idx] == node_id
        cdef:
            SIZE_t j

        # stats_left[0] contains sum_ysq_left
        # stats_left += 1 advances it to the set of memory which stores the
        # sum_y for each output
        stats_left += 1
        stats_right += 1

        # Compute and store the proxies for children impurities for this node
        this_record[0].proxy_impurity_left = 0
        this_record[0].proxy_impurity_right = 0

        for j in range(self.n_outputs):
            this_record[0].proxy_impurity_left -= stats_left[j] ** 2
            this_record[0].proxy_impurity_right -= stats_right[j] ** 2

        # Next, using the left and right proxy impurity, compute the proxy
        # improvement; Minimizing on this, minimizes on the exact mse impurity
        # improvement
        this_record[0].proxy_improvement = (
            this_record[0].proxy_impurity_left /
            this_record[0].weighted_n_left +
            this_record[0].proxy_impurity_right /
            this_record[0].weighted_n_right)
        return 0

    cdef int find_best_split(self, DTYPE_t[:, :] X,
                             DOUBLE_t[:, :] y,
                             double[:] sample_weight,
                             SIZE_t[:, :] X_idx_sorted,
                             SIZE_t[:] shuffled_features,
                             SIZE_t n_features,
                             StoppingCriteria stopping) nogil except -1:
        cdef:
            SIZE_t min_samples_split = stopping.min_samples_split
            SIZE_t min_samples_leaf = stopping.min_samples_leaf
            SIZE_t min_weight_leaf = stopping.min_weight_leaf
            double min_weight_fraction_leaf = stopping.min_weight_fraction_leaf
            SIZE_t max_leaf_nodes = stopping.max_leaf_nodes
            double min_impurity_decrease = stopping.min_impurity_decrease
            double min_impurity_split = stopping.min_impurity_split

            SIZE_t* samples_to_node_idx_map = self.samples_to_node_idx_map

            SIZE_t i, j, k, f, feat_i

            SIZE_t sample_idx, node_idx, node_id
            # x value for i-th sample and f-th feature
            DTYPE_t x_if, x_if_prev
            # y value for i-th sample and k-th output
            double y_ik, w

            bint valid_split
            SplitRecord* this_record

            double* left_stats
            double* right_stats

        # For each feature that is marked to be searched in this job
        for f in range(n_features):
            # Make one pass over the data
            feat_i = shuffled_features[f]
            for i in range(X_idx_sorted.shape[0]):
                sample_idx = X_idx_sorted[i, feat_i]
                x_if = X[sample_idx, feat_i]

                # Find which sample belongs to which node-id and then to which
                # split record / node_stats array index
                node_idx = samples_to_node_idx_map[sample_idx]
                this_record = self.split_records + node_idx
                left_stats = self.node_stats + self.node_stride * node_idx
                right_stats = left_stats + 2 * self.split_stride

                # Don't try to evaluate the split if this is the first entry into the node
                if this_record[0].prev_pos != INVALID:
                    x_if_prev = X[this_record[0].prev_pos, feat_i]

                    valid_split = this_record[0].weighted_n_right > min_weight_leaf
                    valid_split |= this_record[0].weighted_n_left > min_weight_leaf

                    # Check if current node's split is valid and only if that split was valid
                    # we evaluate this split to see if this is the best split so far...
                    # We evaluate the current split before updating the current sample
                    # as only in the next iteration after updating we will get the
                    # X value to compute the threshold
                    # 0.1 | 0.2
                    #       ^
                    #  We can evaulate the split only here as threshold will be (0.1 + 0.2) / 2
                    #                                                                  ===
                    if valid_split:
                        self.store_one_node_proxy_impurity_improvement(
                            node_idx, this_record, left_stats, right_stats)

                        if this_record[0].proxy_improvement > this_record[0].best_proxy_improvement:
                            this_record[0].best_proxy_improvement = this_record[0].proxy_improvement
                            this_record[0].best_pos = sample_idx
                            this_record[0].best_threshold = (X[this_record[0].prev_pos, feat_i] + x_if) * 0.5
                            this_record[0].best_feature = feat_i

                # Update the stats at this point
                # Sample_weight is either specified by user or considered to be 1
                w = sample_weight[sample_idx]

                for k in range(self.n_outputs):
                    y_ik = y[sample_idx, k]
                    w_y = w * y_ik
                    w_y2 = w_y * y_ik

                    # left_stats[0] = sum_ysq_left
                    left_stats[0] += w_y2
                    right_stats[0] -= w_y2

                    # left_stats[1 + k] = sum(y_k) | y_k in left samples
                    left_stats[1 + k] += w_y
                    right_stats[1 + k] -= w_y

                this_record[0].weighted_n_left += w
                this_record[0].weighted_n_right -= w

                this_record[0].prev_pos = this_record[0].pos
                this_record[0].pos = sample_idx