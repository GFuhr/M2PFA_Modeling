#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
import cython
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

from matrix2D cimport LinearMatrix2D

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef void diffusion(double dx, double dy, double C, \
    int iNx, int iNy, \
    double[:, ::1] v, \
    double[:, ::1] u) nogil noexcept :
    cdef int Nx = iNx-1
    cdef int Ny = iNy-1
    cdef int iy, ix
    cdef double v_c
    cdef double hx = C/(dx*dx)
    cdef double hy = C/(dy*dy)
    with parallel(num_threads=4):
        for iy in prange(1, Ny, schedule='static'):
            for ix in range(1, Nx):
                v_c = 2*v[iy, ix]
                u[iy, ix] += hx*(v[iy, ix+1] + v[iy, ix-1] - v_c) + \
                          hy*(v[iy+1, ix] + v[iy-1, ix] - v_c)




cdef void advection(double dx, double dy, double V, \
    int iNx, int iNy, \
    double[:, ::1] v, \
        double[:, ::1] u) nogil noexcept :
    cdef int Nx = iNx-1
    cdef int Ny = iNy-1
    cdef int iy, ix
    cdef double v_c
    cdef double hx = (V/dx)
    cdef double hy = (V/dy)
    with parallel(num_threads=4):
        for iy in prange(1, Ny, schedule='static'):
            for ix in range(1, Nx):
                v_c = v[iy, ix]
                u[iy, ix] += hx*(v_c - v[iy, ix-1]) + hy*(v_c - v[iy-1, ix])


cdef void boundary(
    double[:, ::1] u) nogil  noexcept :
    cdef int Nx = u.shape[1]
    cdef int Ny = u.shape[0]
    cdef int iy , ix

    # y boundary : periodic
    for ix in range(0, Nx):
        u[0, ix] = u[Ny-3, ix] # Ny-3 was Ny-2
        u[Ny-1, ix] = u[2, ix] # 2 was 1
        #u[0, ix] = -u[2, ix]
        #u[Ny-1, ix] = -u[Ny-3, ix]

    # x boundary : u=0
    for iy in range(0, Ny):
        u[iy, 0] = -u[iy, 2]
        u[iy, Nx-1] = -u[iy, Nx-3]


cdef void null_bc(
    double[:, ::1] u) nogil noexcept :
    """
    function to define to implement the null boundary condition
    :param u: 
    :return: 
    """
    cdef int Nx = u.shape[1]
    cdef int Ny = u.shape[0]
    cdef int iy , ix

    # y boundary
    for ix in range(0, Nx):
        # u[0, ix] = -u[2, ix]
        # u[Ny, ix] = -u[Ny-2, ix]
        u[0, ix] = -u[2, ix]
        u[Ny-1, ix] = -u[Ny-3, ix]

    # x boundary
    for iy in range(0, Ny):
        u[iy, 0] = -u[iy, 2]
        u[iy, Nx-1] = -u[iy, Nx-3]



cdef void time_step(
        double[:, ::1] un, \
            double[:, ::1] rhs, int Nx, int Ny, double C, double V, double dx, double dy) nogil noexcept :

    cdef int idx_x, idx_y


    for idx_y in range(0, Ny):
        for idx_x in range(0, Nx):
            rhs[idx_y, idx_x] = 0.

    if (C>1e-6) or (C< -1e-6):
        diffusion(dx, dy, C, Nx, Ny, un, rhs)
    if (V>1e-6) or (V< -1e-6):
        advection(dx, dy, V, Nx, Ny, un, rhs)



def eule(
    double[:, ::1] rhs, \
    double[:, ::1] un, **kwargs):

    cdef int m = un.shape[0]-1
    cdef int n = un.shape[1]-1
    cdef int idx_x, idx_y
    cdef double dt = kwargs.get('dt')
    cdef double C = kwargs.get('C')
    cdef double V = kwargs.get('V')
    cdef double dx = kwargs.get('dx')
    cdef double dy = kwargs.get('dy')

    time_step(un, rhs, un.shape[0], un.shape[1], C*dt, V*dt, dx, dy)
    with nogil, parallel(num_threads=4):
        for idx_y in prange(1, m, schedule='static'):
            for idx_x in range(1, n):
                un[idx_y, idx_x] += rhs[idx_y, idx_x]

    if kwargs.get('boundary', "default").find("default")>-1:
        boundary(un)
    else:
        null_bc(un)


def euli(LinearMatrix2D matA, \
         double[:, ::1] pp1, \
         double[:, ::1] Field_p, **kwargs):
    cdef int m = Field_p.shape[0]-1
    cdef int n = Field_p.shape[1]-1
    cdef int idx_x, idx_y

    with nogil, parallel(num_threads=4):
        for idx_y in prange(1, m, schedule='static'):
            for idx_x in range(1, n):
                pp1[idx_y, idx_x] = Field_p[idx_y, idx_x]
    matA.solve(Field_p, pp1)

    with nogil, parallel(num_threads=4):
        for idx_y in prange(1, m, schedule='static'):
            for idx_x in range(1, n):
                Field_p[idx_y, idx_x] = pp1[idx_y, idx_x]

    if kwargs.get('boundary', "default").find("default")>-1:
        boundary(Field_p)
    else:
        null_bc(Field_p)


def RK_step(
    double[:, ::1] un, \
    double[:, ::1] u_eval, \
    double[:, ::1] ki, \
    double[:, ::1] yi, **kwargs):

    cdef int m = un.shape[0]-1
    cdef int n = un.shape[1]-1
    cdef int idx_x, idx_y
    cdef double gamma = kwargs.get('gamma', .5)

    time_step(u_eval, ki, un.shape[0], un.shape[1], kwargs.get('C'), kwargs.get('V'), kwargs.get('dx'), kwargs.get('dy') )
    with nogil, parallel(num_threads=4):
        for idx_y in prange(1, m, schedule='static'):
            for idx_x in range(1, n):
                yi[idx_y, idx_x] = un[idx_y, idx_x] + gamma*ki[idx_y, idx_x]

    if kwargs.get('boundary', "default").find("default")>-1:
        boundary(yi)
    else:
        null_bc(yi)


def RK4(
    double[:, ::1] k1, \
    double[:, ::1] k2, \
    double[:, ::1] k3, \
    double[:, ::1] k4, \
    double[:, ::1] y1, \
    double[:, ::1] y2, \
    double[:, ::1] y3, \
    double[:, ::1] Field_p,
             **kwargs):

    cdef int m = Field_p.shape[0]
    cdef int n = Field_p.shape[1]
    cdef int idx_x, idx_y, start = 1
    cdef double dt = kwargs.get('dt')

    k1[:] = 0; k2[:] = 0; k3[:] = 0; k4[:] = 0
    y1[:] = 0; y2[:] = 0; y3[:] = 0

    # y1 = p + dt/2*k1
    # k1 = rhs(p)
    kwargs['gamma'] = .5*dt
    RK_step(Field_p,Field_p, k1, y1, **kwargs)

    # y2 = p + dt/2*k2
    # k2 = rhs(y1)
    kwargs['gamma'] = .5*dt
    RK_step(Field_p, y1, k2, y2, **kwargs)

    # y3 = p + dt*k3
    kwargs['gamma'] = dt
    RK_step(Field_p, y2, k3, y3, **kwargs)

    # k4 = rhs(y3)
    time_step(y3, k4, Field_p.shape[0], Field_p.shape[1], kwargs.get('C'),  kwargs.get('V'), kwargs.get('dx'), kwargs.get('dy'))

    with nogil, parallel(num_threads=4):
        for idx_y in prange(start, m, schedule='static'):
            for idx_x in range(start, n):
                Field_p[idx_y, idx_x] = Field_p[idx_y, idx_x]+\
                                    (dt/6.)*(k1[idx_y, idx_x]+2.*(k2[idx_y, idx_x]+k3[idx_y, idx_x])+k4[idx_y, idx_x])

    if kwargs.get('boundary', "default").find("default")>-1:
        boundary(Field_p)
    else:
        null_bc(Field_p)


def RK2(
    double[:, ::1] k1, \
    double[:, ::1] k2, \
    double[:, ::1] y1, \
    double[:, ::1] Field_p,
             **kwargs):

    cdef int m = Field_p.shape[0]
    cdef int n = Field_p.shape[1]
    cdef int idx_x, idx_y, start = 1
    cdef double dt = kwargs.get('dt')


    k1[:,:] = 0; k2[:,:] = 0
    y1[:,:] = 0

    # k1 = rhs(p), y1 = p + dt/2*k1  (midpoint)
    kwargs['gamma'] = .5*dt
    RK_step(Field_p, Field_p, k1, y1, **kwargs)
    
    # k2 = rhs(y1)
    time_step(y1, k2, Field_p.shape[0], Field_p.shape[1], kwargs.get('C'), kwargs.get('V'), kwargs.get('dx'), kwargs.get('dy'))

    with nogil, parallel(num_threads=4):
        for idx_y in prange(start, m, schedule='static'):
            for idx_x in range(start, n):
                Field_p[idx_y, idx_x] = Field_p[idx_y, idx_x] + (dt)*k2[idx_y, idx_x]

    if kwargs.get('boundary', "default").find("default")>-1:
        boundary(Field_p)
    else:
        null_bc(Field_p)

