import os.path

def datapath(fname):
    basedir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(basedir, fname)
