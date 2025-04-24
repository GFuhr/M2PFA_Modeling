#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Guillaume Fuhr"

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from advdiff.python_vector import (MatrixSolver, eule, euli, RK2, RK4, CranckN)
from line_profiler import profile

sys.path.append(os.path.normpath(os.path.join(os.path.realpath(__file__), "../..")))

try:
    from parameters import load_params, initfield_1d
except ModuleNotFoundError:
    from advdiff.parameters import load_params, initfield_1d


@profile
def simulate(verbose=False, save_files=False, init=None, **kwargs):
    """Vectorized numpy implementation of the simulation"""
    global_params = load_params(**kwargs)
    Nx = global_params["Nx"]
    dx = global_params["dx"]
    dt = global_params["dt"]
    Tmax = global_params["Tmax"]
    Toutput = global_params["Toutput"]
    C = global_params["C"]
    V = global_params["V"]

    # Initialize fields and constants using numpy arrays
    X = np.linspace(0, (Nx-1)*dx, Nx)
    
    init_u0 = init or initfield_1d
    Field_p_init = init_u0(X)

    # RK Fields
    k1 = np.zeros(Nx)
    k2 = np.zeros(Nx)
    k3 = np.zeros(Nx)
    k4 = np.zeros(Nx)
    y1 = np.zeros(Nx)
    y2 = np.zeros(Nx)
    y3 = np.zeros(Nx)

    # Matrix for implicit schemes
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

    # Setup derivative arrays using numpy arrays
    derivative = global_params["derivative"].strip().lower()
    darray = np.zeros(3)
    if derivative == "fwd":
        darray = np.array([0, -1, 1], dtype=np.float64)
    elif derivative == "bwd":
        darray = np.array([-1, 1, 0], dtype=np.float64)
    elif derivative == "cent":
        darray = np.array([-1, 0, 1], dtype=np.float64)
    else:
        raise ValueError("derivative can be : fwd, bwd or cent")
    
    darray /= (1.0*dx)
    vector_op = (global_params["V"] * darray + 
                global_params["C"] * np.array([1, -2, 1]) / (dx * dx))

    # Setup boundary conditions
    boundaries = global_params.get("boundaries", "dir").strip().lower()
    if boundaries == "dir":
        bc = np.array([-1, 0, -1, 0])
    elif boundaries == "neu":
        bc = np.array([1, 0, 1, 0])
    elif boundaries == "per":
        bc = np.array([0, 1, 0, 1])
    else:
        raise ValueError("boundary conditions not defined")
    global_params["bc"] = bc

    Field_p = Field_p_init.copy()
    Frames = [Field_p_init.copy()]
    TimeOutput = [0]

    formatted_scheme = global_params["scheme"].strip().lower()
    if formatted_scheme == "cn":
        vector_op *= 0.5
    
    Mat.create_matrix(np.array([0, 1, 0])-vector_op, bc)

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
    plt.clf()
    plt.plot(X, Y)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.pause(0.01)

if __name__ == "__main__":
    # Run simulation
    TimeOutput, Frames, _ = simulate(verbose=True)
    
    # Plot results
    X = np.linspace(0, 1, len(Frames[0]))
    plt.ion()
    for frame in Frames:
        plot_frame(X, frame)
    plt.ioff()
    plt.show() 