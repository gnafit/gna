# -*- coding: utf-8 -*-

from os.path import splitext
import ROOT as R
from tools.root_helpers import TFileContext
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
import numpy as np

def read_object_root(filename, name, **kwargs):
    with TFileContext(filename) as f:
        o = f.Get( name )
        if not o:
            raise IOError("Can not read object `{}` from file `{}`"%(name, filename) )

        return get_buffers_graph_or_hist1(o)

def read_dat(filename, **kwargs):
    return np.loadtxt(filename, unpack=True)

# def read_object_hdf5(filename, names, *args, **kwargs):
    # pass

def read_dat(filename, **kwargs):
    return np.load(filename, unpack=True)

readers = {
        '.root' : read_object_root,
        # '.hdf5' : read_object_hdf5,
        '.npz'  : read_object_npz,
        '.dat'  : read_dat,
        '.txt'  : read_dat,
        }

def read_object_auto(filename, **kwargs):
    """Load an object from npz/hdf5/ROOT file"""
    verbose = kwargs.pop('verbose', False)
    name = kwargs.get('name')
    if verbose:
        if name:
            print('Read {}: {}{}'.format(filename, name, suffix))
        else:
            print('Read {} {}'.format(filename, suffix))

    ext = splitext(filename)[-1]
    reader = readers.get(ext, read_dat)

    try:
        return reader(filename, **kwargs)
    except:
        raise Exception('Unable to read {}')


