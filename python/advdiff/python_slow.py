from .bicgstab_python import MatrixSolver as MatrixSolver

def op_copy(dest, src):
    for i in range(len(dest)):
        dest[i] = src[i]

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

def eule(rhs, field_p, adv_factor, bc):
    time_step(field_p, rhs, adv_factor)
    op_axpby(rhs, 1, field_p, 1)
    boundary(field_p, bc)

def euli(matA, pp1, field_p, bc):
    matA.solve(field_p, pp1)
    op_copy(field_p, pp1)
    boundary(field_p, bc)

def CranckN(matA, rhs, field_p, adv_factor, bc):
    eule(rhs, field_p, adv_factor, bc)
    euli(matA, rhs, field_p, bc)

def RK_step(field_p, ki, ppi, adv_factor, bc, gamma):
    time_step(field_p, ki, adv_factor)
    op_copy(ppi, field_p)
    op_axpby(ki, gamma, ppi, 1)
    boundary(ppi, bc)

def RK4(k1, k2, k3, k4, y1, y2, y3, field_p, adv_factor, bc):
    RK_step(field_p, k1, y1, adv_factor, bc, 0.5)
    RK_step(y1, k2, y2, adv_factor, bc, 0.5)
    RK_step(y2, k3, y3, adv_factor, bc, 1.0)
    time_step(y3, k4, adv_factor)
    
    # Combine steps
    for i in range(len(k3)):
        k3[i] = (k2[i] + k3[i]) / 3.0
    for i in range(len(k4)):
        k4[i] = (k1[i] + k4[i]) / 6.0
    op_axpby(k3, 1, k4, 1)
    op_axpby(k4, 1, field_p, 1)
    boundary(field_p, bc)

def RK2(k1, k2, y1, field_p, adv_factor, bc):
    RK_step(field_p, k1, y1, adv_factor, bc, 0.5)
    time_step(y1, k2, adv_factor)
    op_axpby(k2, 1, field_p, 1)
    boundary(field_p, bc) 

def op_zero(x):
    for i in range(len(x)):
        x[i] = 0.0

def diffusion(dx, c, v, u):
    hx = c/(dx*dx)
    for i in range(1, len(u)-1):
        u[i] += hx * (v[i+1] + v[i-1] - 2*v[i])

def diffusion_coef(dx, Lx, chi, dchi, v, u):
    hx = 1/(dx*dx)
    lxm2 = 1./(Lx*Lx)
    for i in range(1, len(u)-1):
        u[i] += lxm2 * (
            chi[i] * hx * (v[i+1] + v[i-1] - 2*v[i]) + 
            dchi[i] * (v[i+1] - v[i-1]) * 0.5/dx
        )

def advection(dx, adv_factor, v, u):
    for i in range(1, len(u)-1):
        u[i] += (adv_factor[0]*v[i-1] + 
                 adv_factor[1]*v[i] + 
                 adv_factor[2]*v[i+1])

def advection_diffusion(adv_factor, v, u):
    for i in range(1, len(u)-1):
        u[i] = (adv_factor[0]*v[i-1] + 
                adv_factor[1]*v[i] + 
                adv_factor[2]*v[i+1])
