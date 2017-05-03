# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from sklearn.utils.testing import assert_almost_equal


cpdef check_mse_unweighted_single_output():
    cdef:
        Node* nodes
        double* node_stats
        SIZE_t node_stats_stride

    mse = MSE(nodes, node_stats, 100)

    # The number of stats that we will store for MSE
    node_stats_stride = mse.n_stats
    safe_realloc(&nodes, 3)
    safe_realloc(&node_stats, 3 * node_stats_stride)

    nodes[0].weighted_n_node_samples = 100
    # Let's keep first 70 samples in the right and the next 30 in the left
    nodes[1].weighted_n_node_samples = 70
    nodes[2].weighted_n_node_samples = 30

    # Populate the node_stats with relevant information

    rng = np.random.RandomState(0)
    y = rng.random_sample((100, 1))

    # For root
    node_stats[0] = y.mean()
    node_stats[1] = y.sum()
    node_stats[2] = np.sum(y ** 2)

    # For left child
    y_left = y[:70]
    node_stats[3] = y_left.mean()
    node_stats[4] = y_left.sum()
    node_stats[5] = np.sum(y_left ** 2)

    # For right child
    y_right = y[70:]
    node_stats[6] = y_right.mean()
    node_stats[7] = y_right.sum()
    node_stats[8] = np.sum(y_right ** 2)

    # Check that the variance for the root node is correct
    assert_almost_equal(mse._impurity(0), 5.0)
    assert_almost_equal(mse._impurity(1), 5.0)
    assert_almost_equal(mse._impurity(2), 5.0)

    free(nodes)
    free(node_stats)