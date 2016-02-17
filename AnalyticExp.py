import numpy as np

def calculate_cycloid_radius(lx, ly):
    pass

def generate_cycloid(lx, ly, numpts=101):
    r = calculate_cycloid_radius(lx, ly)
    t = np.linspace(0, np.arccos(1.-ly/r), numpts)
    x = r * (t-np.sin(t))
    y = ly - r * (1-np.cos(t))
    return x, y