# -*- coding: utf-8 -*-

from os.path import splitext

def read_object_root( filename, name, *args, **kwargs ):
    import ROOT as R
    f = R.TFile( filename, 'READ' )
    if f.IsZombie():
        raise IOError( 'Can not read ROOT file: '+filename )

    o = f.Get( name )
    if not o:
        raise IOError( "Can not read object '%s' from file '%s'"%( name, filename ) )

    fmt = kwargs.pop( 'convertto', None )
    if fmt:
        from converters import convert
        return convert(o, fmt)

    return o

def read_object_hdf5( filename, names, *args, **kwargs ):
    pass

def read_object_npz( filename, names, *args, **kwargs ):
    pass

readers = {
        '.root' : read_object_root,
        '.hdf5' : read_object_hdf5,
        '.npz'  : read_object_npz
        }

def read_object_auto( filename, *args, **kwargs ):
    """Load an object from npz/hdf5/ROOT file"""
    ext = splitext(filename)[-1]
    if not ext in readers:
        raise Exception( "Can not read file with extension '%s'"%ext )

    return readers[ext]( filename, *args, **kwargs )


