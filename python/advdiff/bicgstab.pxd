#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: warn.undeclared=True

cdef class MatrixSolver(object):
    cdef Py_ssize_t size
    cdef double[::1] dummy
    cdef double[::1]  r_i
    cdef double[::1]  s_i
    cdef double[::1]  v_i
    cdef double[::1]  p_i
    cdef double[::1]  t_i
    cdef double[::1]  r_hat_0
    cdef double[:,::1]  operator
    cdef bint isinit
    cdef double tol
    cdef int iter
    cdef init_variables(self) noexcept
    cpdef double[:,::1] create_matrix(self, const double[::1] vector_op,
                                      const long[::1] bc) noexcept
    cpdef void solve(self, const double[::1] rhs, double[::1] sol) noexcept
    
