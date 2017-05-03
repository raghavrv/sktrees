# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters

from sklearn.tree._tree cimport Node
from sklearn.tree._utils cimport safe_realloc
from sklearn.fast_tree._criterion cimport MSE
