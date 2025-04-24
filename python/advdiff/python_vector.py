import numpy as np
from .bicgstab_python import MatrixSolver as MatrixSolver
# class MatrixSolver:
#     def __init__(self, size):
#         self.size = size - 2
#         self.tol = 1e-7
#         self.iter = 1000
#         self.isinit = False
#         self.init_variables()

#     def init_variables(self):
#         self.isinit = True
#         # Initialize arrays with zeros using numpy
#         self.dummy = np.zeros(self.size)
#         self.r_i = np.zeros(self.size)
#         self.s_i = np.zeros(self.size)
#         self.v_i = np.zeros(self.size)
#         self.p_i = np.zeros(self.size)
#         self.t_i = np.zeros(self.size)
#         self.r_hat_0 = np.zeros(self.size)
#         self.matrix_op = np.zeros((3, 5))

#     def create_matrix(self, vector_op, bc):
#         if not self.isinit:
#             self.init_variables()

#         # Vectorized matrix creation
#         self.matrix_op[0, 1:4] = vector_op
#         self.matrix_op[1, 1:4] = vector_op
#         self.matrix_op[2, 1:4] = vector_op

#         # Handle boundary conditions
#         self.matrix_op[0, 3] += bc[0] * vector_op[0]
#         self.matrix_op[0, 1] = 0
#         self.matrix_op[0, 4] = bc[1] * vector_op[0]

#         self.matrix_op[2, 1] += bc[2] * vector_op[2]
#         self.matrix_op[2, 3] = 0
#         self.matrix_op[2, 0] = bc[3] * vector_op[2]
        
#         return self.matrix_op

#     def solve(self, rhs, x):
#         x[1:-1] = rhs[1:-1]

# def op_xpy(x, y):
#     np.add(x, y, out=x)

# def op_copy(dest, src):
#     np.copyto(dest, src)

def op_axpby(x, a, y, b):
    y[:] = a * x + b * y

def time_step(field_p, pp1, adv_factor):
    pp1[1:-1] = (adv_factor[0] * field_p[:-2] + 
                 adv_factor[1] * field_p[1:-1] + 
                 adv_factor[2] * field_p[2:])

def boundary(u, bc):
    m = len(u) - 1
    u[0] = bc[0] * u[2] + bc[1] * u[m-1]
    u[m] = bc[2] * u[m-2] + bc[3] * u[1]

def op_zero(x):
    x.fill(0)

def triband_dot(band_matrix, x, vec, matrix_size):
    # First row special case
    vec[0] = np.dot(band_matrix[0, 2:5], 
                    np.array([x[0], x[1], x[matrix_size-1]]))
    
    # Middle rows - vectorized
    vec[1:-1] = (band_matrix[1, 1] * x[:-2] + 
                 band_matrix[1, 2] * x[1:-1] + 
                 band_matrix[1, 3] * x[2:])
    
    # Last row special case
    vec[-1] = (band_matrix[2, 1] * x[-2] + 
               band_matrix[2, 2] * x[-1] + 
               band_matrix[2, 0] * x[0])

def eule(rhs, field_p, adv_factor, bc):
    time_step(field_p, rhs, adv_factor)
    field_p[:] += rhs
    boundary(field_p, bc)

def euli(matA, pp1, field_p, bc):
    matA.solve(field_p, pp1)
    np.copyto(field_p, pp1)
    boundary(field_p, bc)

def CranckN(matA, rhs, field_p, adv_factor, bc):
    eule(rhs, field_p, adv_factor, bc)
    euli(matA, rhs, field_p, bc)

def RK_step(field_p, ki, ppi, adv_factor, bc, gamma):
    time_step(field_p, ki, adv_factor)
    np.copyto(ppi, field_p)
    ppi[:] += gamma * ki
    boundary(ppi, bc)

def RK4(k1, k2, k3, k4, y1, y2, y3, field_p, adv_factor, bc):
    RK_step(field_p, k1, y1, adv_factor, bc, 0.5)
    RK_step(y1, k2, y2, adv_factor, bc, 0.5)
    RK_step(y2, k3, y3, adv_factor, bc, 1.0)
    time_step(y3, k4, adv_factor)
    
    # Vectorized combination of steps
    k3[:] = (k2 + k3) / 3.0
    k4[:] = (k1 + k4) / 6.0
    k4[:] += k3
    field_p[:] += k4
    boundary(field_p, bc)

def RK2(k1, k2, y1, field_p, adv_factor, bc):
    RK_step(field_p, k1, y1, adv_factor, bc, 0.5)
    time_step(y1, k2, adv_factor)
    field_p[:] += k2
    boundary(field_p, bc) 