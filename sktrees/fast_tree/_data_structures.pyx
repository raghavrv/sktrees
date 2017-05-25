# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
#
# License: BSD 3 clause


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


cdef class SplitRecordStore:
    """This class maintains a set of SplitRecord structs for use by the tree builder"""
    def __cinit__(self, n_records):
        self.array = NULL
        safe_realloc(&self.array, <size_t> n_records)
        self.n_records = n_records

    cdef int _resize(self, size_t n_records) except -1:
        # Always resize to twice the required number of records

        # Downsize to `2 * n_records` only if *more* than half the current
        # space is unused
        if self.n_records > 2 * n_records:
            self.n_records = 2 * n_records

        # Or if the number of required records is more than what we have
        elif self.n_records <= n_records:
            self.n_records = 2 * n_records

        else:
            return 0

        safe_realloc(&self.array, self.n_records)
        return 0

    def __dealloc__(self):
        free(self.array)

    cdef np.ndarray _get_split_record_ndarray(self):
        cdef SIZE_t shape[1]
        cdef SIZE_t strides[1]
        shape[0] = <SIZE_t> self.n_records
        strides[0] = SplitRecord_SIZE_BYTES
        cdef np.ndarray arr

        # We say to the python's garbage collector that
        # we have created a copy of the SPLITRECORD_NP_STRUCT_DTYPE
        # python object and we will be retaining it inside `arr`
        Py_INCREF(SPLITRECORD_NP_STRUCT_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> SPLITRECORD_NP_STRUCT_DTYPE,
                                   1, shape, strides,
                                   <void*> self.array,
                                   np.NPY_DEFAULT, None)
        # And here too as we give a reference of self to arr as `arr.base`
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr