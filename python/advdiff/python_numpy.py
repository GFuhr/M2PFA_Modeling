import numpy as np
from .bicgstab_numpy import MatrixSolver as MatrixSolver


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

#         # Copy vector_op values to matrix_op
#         for i in range(3):
#             self.matrix_op[0, i+1] = vector_op[i]

#         # Handle boundary conditions
#         self.matrix_op[0, 3] += bc[0] * vector_op[0]
#         self.matrix_op[0, 1] = 0
#         self.matrix_op[0, 4] = bc[1] * vector_op[0]

#         for i in range(3):
#             self.matrix_op[1, i+1] = vector_op[i]

#         for i in range(3):
#             self.matrix_op[2, i+1] = vector_op[i]

#         self.matrix_op[2, 1] += bc[2] * vector_op[2]
#         self.matrix_op[2, 3] = 0
#         self.matrix_op[2, 0] = bc[3] * vector_op[2]
#         return self.matrix_op

#     def solve(self, rhs, x):
#         for i in range(1, len(x)-1):
#             x[i] = rhs[i]

# def op_xpy(x, y):
#     for i in range(len(x)):
#         x[i] += y[i]

# def op_copy(dest, src):
#     for i in range(len(dest)):
#         dest[i] = src[i]

def op_axpby(x, a, y, b):
    for i in range(len(y)):
        y[i] = a * x[i] + b * y[i]

def time_step(field_p, pp1, adv_factor):
    for i in range(1, len(field_p)-1):
        pp1[i] = (adv_factor[0] * field_p[i-1] + 
                 adv_factor[1] * field_p[i] + 
                 adv_factor[2] * field_p[i+1])

def boundary(u, bc):
    m = len(u) - 1
    u[0] = bc[0] * u[2] + bc[1] * u[m-1]
    u[m] = bc[2] * u[m-2] + bc[3] * u[1]

# The rest of the functions remain the same but using numpy arrays
# Only showing the modified functions for brevity

def op_zero(x):
    for i in range(len(x)):
        x[i] = 0.0

def triband_dot(band_matrix, x, vec, matrix_size):
    # First row
    vec[0] = (band_matrix[0, 2]*x[0] + 
              band_matrix[0, 3]*x[1] + 
              band_matrix[0, 4]*x[matrix_size-1])
    
    # Middle rows
    bmm1 = band_matrix[1, 1]
    bm = band_matrix[1, 2]
    bmp1 = band_matrix[1, 3]
    for i in range(1, matrix_size-1):
        vec[i] = bmm1*x[i-1] + bm*x[i] + bmp1*x[i+1]
    
    # Last row
    i = matrix_size-1
    vec[i] = (band_matrix[2, 1]*x[i-1] + 
              band_matrix[2, 2]*x[i] + 
              band_matrix[2, 0]*x[0]) 

def diffusion(dx, c, v, u):
    hx = c/(dx*dx)
    u[1:-1] += hx * (v[2:] + v[:-2] - 2*v[1:-1])

def diffusion_coef(dx, Lx, chi, dchi, v, u):
    hx = 1/(dx*dx)
    lxm2 = 1./(Lx*Lx)
    u[1:-1] += lxm2 * (
        chi[1:-1] * hx * (v[2:] + v[:-2] - 2*v[1:-1]) + 
        dchi[1:-1] * (v[2:] - v[:-2]) * 0.5/dx
    )

def advection(dx, adv_factor, v, u):
    u[1:-1] += (adv_factor[0]*v[:-2] + 
                adv_factor[1]*v[1:-1] + 
                adv_factor[2]*v[2:])

def advection_diffusion(adv_factor, v, u):
    u[1:-1] = (adv_factor[0]*v[:-2] + 
               adv_factor[1]*v[1:-1] + 
               adv_factor[2]*v[2:])

def op_xpby(x, y, b):
    x[:] += b * y

def d_axpbypcz(vec_size, x, a, y, b, z, c):
    z[:] = a*x + b*y + c*z

def vec_dot(x, y, vec_size):
    return np.dot(x[:vec_size], y[:vec_size])

def single_vec_dot(x, vec_size):
    return np.dot(x[:vec_size], x[:vec_size])

def vec_xmy(res, x, y, vec_size):
    res[:vec_size] = x[:vec_size] - y[:vec_size]

def dcopy(vec_size, src, dest):
    dest[:vec_size] = src[:vec_size]

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
    
    # Combine steps
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