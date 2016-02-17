import numpy as np
from scipy.optimize import fsolve

def calculate_cycloid_radius(lx, ly):
    f = lambda xi: np.arccos(1-xi) - np.sqrt(1-(1-xi)**2) - lx/ly*xi
    return ly/fsolve(f, 1.2)

def generate_cycloid(lx, ly, numpts=101):
    r = calculate_cycloid_radius(lx, ly)
    t = np.linspace(0, np.arccos(1.-ly/r), numpts)
    x = r * (t-np.sin(t))
    y = ly - r * (1-np.cos(t))
    return x, y