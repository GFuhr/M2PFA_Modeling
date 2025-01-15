# distutils: language = c++
# cython: language_level=3

cimport numpy as cnp
import numpy as np

cdef class MatrixSolver:
    cdef:
        public Py_ssize_t size
        public double tol
        public int iter
        public bint isinit
        double[:,::1] operator
        double[::1] dummy, r_i, s_i, v_i, p_i, t_i, r_hat_0
        
    cdef void init_variables(self) noexcept
    cpdef double[:,::1] create_matrix(self, const double[::1] vector_op, const long[::1] bc) noexcept
    cpdef void solve(self, const double[::1] rhs, double[::1] sol) noexcept
