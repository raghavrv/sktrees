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


cdef struct Split:
    DOUBLE_t a
    SIZE_t b


ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (Split*)
    (Split**)
    (SIZE_t*)
    (DOUBLE_t*)


cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError("could not allocate (%d * %d) bytes"
                          % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


# We make a numpy struct dtype for converting the array of Split structs
# to a python numpy array
SPLIT_NP_STRUCT_DTYPE = np.dtype({
    'names': ['a', 'b'],
    'formats': [np.float64, np.intp],
    'offsets': [
        <Py_ssize_t> &((<Split*> NULL).a),
        <Py_ssize_t> &((<Split*> NULL).b)
    ]
})


cdef class SplitRecords:
    """This is to help passing the array as a python object in joblib"""
    cdef Split** array
    cdef SIZE_t n_records

    def __cinit__(self, n_records):
        self.array = NULL
        safe_realloc(&self.array, <size_t> n_records)
        self.n_records = n_records
    
    def _resize(self, size_t n_records):
        # Downsize only if half of the space is unused
        if (self.n_records > 2 * n_records) or (self.n_records < n_records):
            safe_realloc(&self.array, <size_t> n_records)
            self.n_records = n_records

    def __dealloc__(self):
        free(self.array)

    cdef np.ndarray _get_split_record_ndarray(self):
        cdef SIZE_t shape[1], strides[1]
        shape[0] = <SIZE_t> self.n_records
        strides[0] = sizeof(Split)
        cdef np.ndarray arr
        
        # We say to the python's garbage collector that
        # we have created a copy of the SPLITRECORD_NP_STRUCT_DTYPE
        # python object and we will be retaining it inside `arr`
        Py_INCREF(SPLIT_NP_STRUCT_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> SPLIT_NP_STRUCT_DTYPE,
                                   1, shape, strides,
                                   <void*> self.array,
                                   np.NPY_DEFAULT, None)
        # Why??
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr


r = SplitRecords(10)

cpdef bar(SplitRecords records, size_t i):
    cdef Split* r = (<Split*> records.array) + i
    r.a = i / 10.
    r.b = i * 10
    
    (r + 1).a = 0.5
    (r + 1).b = 50
    return records._get_split_record_ndarray()

a = bar(r, 0) # 0, 1

a['a'][2] = 0.8
a['b'][2] = 80

a = bar(r, 3)