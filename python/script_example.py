# -*- coding: utf-8 -*-
__author__ = "Guillaume Fuhr"

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
try:
    from utils.plotting import figformat, animated_plot_2d, animated_plot_1d
except ModuleNotFoundError:
    sys.path.append(os.path.normpath(os.path.realpath(__file__)))
    from utils.plotting import figformat, animated_plot_2d, animated_plot_1d

# Simple pyximport setup without cleanup
import pyximport
pyximport.install(setup_args={
    'include_dirs': [np.get_include(), './advdiff'],
})

if int(sys.version[0]) != 3:
    sys.exit("This script requires Python 3")

import advdiff

if __name__ == '__main__':
    tout, data1d, _ = advdiff.simulate_1d()
    anim = animated_plot_1d(data1d)
    plt.plot(data1d[-1])
    plt.show()
