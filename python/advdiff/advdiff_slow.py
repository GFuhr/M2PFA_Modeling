#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Guillaume Fuhr"

import sys
import os
import time
import matplotlib.pyplot as plt
from advdiff.python_slow import (MatrixSolver, eule, euli, RK2, RK4, CranckN)
from line_profiler import profile

sys.path.append(os.path.normpath(os.path.join(os.path.realpath(__file__), "../..")))

try:
    from parameters import load_params, initfield_1d
except ModuleNotFoundError:
    from advdiff.parameters import load_params, initfield_1d

def linspace(start, stop, num):
    """Pure Python implementation of numpy.linspace"""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]



def array_multiply(arr, scalar):
    """Multiply each element of array by scalar"""
    return [x * scalar for x in arr]

@profile
def simulate(verbose=False, save_files=False, init=None, **kwargs):
    """Pure Python implementation of the simulation"""
    global_params = load_params(**kwargs)
    Nx = global_params["Nx"]
    dx = global_params["dx"]
    dt = global_params["dt"]
    Tmax = global_params["Tmax"]
    Toutput = global_params["Toutput"]
    C = global_params["C"]
    V = global_params["V"]

    # init fields and constants
    X = [dx * x for x in range(Nx)]
    
    init_u0 = init or initfield_1d
    try:
        Field_p_init = init_u0(X)
    except TypeError:
        import numpy as np
        Field_p_init = init_u0(np.array(X))
        Field_p_init = [float(x) for x in Field_p_init]

    # RK Fields
    k1 = [0.0] * Nx
    k2 = [0.0] * Nx
    k3 = [0.0] * Nx
    k4 = [0.0] * Nx
    y1 = [0.0] * Nx
    y2 = [0.0] * Nx
    y3 = [0.0] * Nx

    # matrix for implicit schemes
    Mat = MatrixSolver(Nx)

    if verbose:
        print("stability numbers : ")
        print("advection : {0}".format(abs(V) * dt / dx))
        print("diffusion : {0}".format(C * dt / dx ** 2))
        print("scheme : {0}".format(global_params["scheme"]))
        print("boundaries : {0}".format(global_params["boundaries"]))

    t = 0
    tlast = 0
    global_params["C"] *= dt
    global_params["V"] *= dt

    # Setup derivative arrays
    derivative = global_params["derivative"].strip().lower()
    if derivative == "fwd":
        darray = [0, -1, 1]
    elif derivative == "bwd":
        darray = [-1, 1, 0]
    elif derivative == "cent":
        darray = [-1, 0, 1]
    else:
        raise ValueError("derivative can be : fwd, bwd or cent")
    
    # Scale darray
    darray = array_multiply(darray, 1/dx)

    # Create vector_op
    vector_op = [0.0] * 3
    diffusion_terms = [1, -2, 1]
    for i in range(3):
        vector_op[i] = (global_params["V"] * darray[i] + 
                        global_params["C"] * diffusion_terms[i] / (dx * dx))

    # Setup boundary conditions
    boundaries = global_params.get("boundaries", "dir").strip().lower()
    if boundaries == "dir":
        bc = [-1, 0, -1, 0]
    elif boundaries == "neu":
        bc = [1, 0, 1, 0]
    elif boundaries == "per":
        bc = [0, 1, 0, 1]
    else:
        raise ValueError("boundary conditions not defined")
    global_params["bc"] = bc

    Field_p = Field_p_init.copy()
    Frames = [Field_p_init.copy()]
    TimeOutput = [0]

    formatted_scheme = global_params["scheme"].strip().lower()
    if formatted_scheme == "cn":
        vector_op = array_multiply(vector_op, 0.5)
    
    # Create vector_mat explicitly
    vector_mat = [0.0] * 3
    for i in range(3):
        vector_mat[i] = (1.0 if i == 1 else 0.0) - vector_op[i]

    Mat.create_matrix(vector_mat, bc)

    iterations = 0
    start_time = time.time()

    while t < Tmax:
        if formatted_scheme == "eule":
            eule(k1, Field_p, vector_op, bc)
        elif formatted_scheme == "euli":
            euli(Mat, k1, Field_p, bc)
        elif formatted_scheme == "rk2":
            RK2(k1, k2, y1, Field_p, vector_op, bc)
        elif formatted_scheme == "rk4":
            RK4(k1, k2, k3, k4, y1, y2, y3, Field_p, vector_op, bc)
        elif formatted_scheme == "cn":
            CranckN(Mat, k1, Field_p, vector_op, bc)
        else:
            raise ValueError("scheme not specified")

        if (t - tlast) > Toutput:
            if verbose:
                print('processing.... {0}%'.format(int(100. * t / Tmax)))
            tlast += Toutput
            Frames.append(Field_p.copy())
            TimeOutput.append(t + dt)
        t += dt
        iterations += 1

    execution_time = (time.time() - start_time) * 1e6  # Convert to microseconds
    print("total execution time in µs : {0}".format(execution_time))
    print("number of snapshots : {0}".format(len(Frames)))
    print("used time for 1 time step : {0:.2f} µs".format(execution_time / iterations))

    return TimeOutput, Frames, execution_time / iterations

def plot_frame(X, Y):
    """Simple plotting function without numpy dependency"""
    plt.clf()
    plt.plot(X, Y)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.pause(0.01)

if __name__ == "__main__":
    # Run simulation
    TimeOutput, Frames, _ = simulate(verbose=True)
    
    # Plot results
    X = linspace(0, 1, len(Frames[0]))
    plt.ion()
    for frame in Frames:
        plot_frame(X, frame)
    plt.ioff()
    plt.show() 