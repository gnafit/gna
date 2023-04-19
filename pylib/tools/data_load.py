from os.path import splitext
import ROOT as R
from tools.root_helpers import TFileContext
from mpl_tools.root2numpy import get_buffers_auto
import numpy as np

def read_root(filename, name, **kwargs):
    if not name:
        raise ValueError('A name should be provided to read an object from a ROOT file')

    with TFileContext(filename) as f:
        o = f.Get( name )
        if not o:
            raise IOError("Can not read object `{}` from file `{}`"%(name, filename) )

        return tuple(o.copy() for o in get_buffers_auto(o))

def read_dat(filename, **kwargs):
    kwargs.setdefault('unpack', True)
    return np.loadtxt(filename, **kwargs)

# def read_object_hdf5(filename, names, *args, **kwargs):
    # pass

def read_npz(filename, name=None, **kwargs):
    ret = np.load(filename)

    if name:
        return ret[name]

    return ret

readers = {
        '.root' : read_root,
        # '.hdf5' : read_object_hdf5,
        '.npz'  : read_npz,
        '.dat'  : read_dat,
        '.txt'  : read_dat,
        }

def read_object_auto(filename, **kwargs):
    """Load an object from npz/hdf5/ROOT file"""
    verbose = kwargs.pop('verbose', False)
    suffix = kwargs.pop('suffix', '')
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
        print('Unable to read {} ({})'.format(filename, str(kwargs)))
        raise
