import numpy as np
from scipy.optimize import bisect

def calculate_cycloid_radius(lx, ly):
    f = lambda r: r*(np.cos(1-ly/r)-np.sqrt(1-(1-ly/r)**2))-lx
    return bisect(f, 0.5*ly, ly)

def generate_cycloid(lx, ly, numpts=101):
    r = calculate_cycloid_radius(lx, ly)
    t = np.linspace(0, 2*np.pi-np.arccos(1.-ly/r), numpts)
    x = r * (t-np.sin(t))
    y = ly - r * (1-np.cos(t))
    return x, y