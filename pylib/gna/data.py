import os.path
import numpy as np

def load(fname):
    basedir = os.path.join(os.path.dirname(__file__), "../../data")
    return np.loadtxt(os.path.join(basedir, fname))
