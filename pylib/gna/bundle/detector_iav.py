#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R

def read_object_root( filename, name ):
    f = R.TFile( filename, 'READ' )
    if f.IsZombie():
        raise IOError( 'Can not read ROOT file: '+filename )

    o = f.Get( name )
    if not o:
        raise IOError( "Can not read object '%s' from file '%s'"%( name, filename ) )

    return o

def read_object_hdf5( filename, name ):
    pass
    # f = R.TFile( filename, 'READ' )
    # if f.IsZombie():
        # raise IOError( 'Can not read ROOT file: '+filename )

    # o = f.Get( name )
    # if not o:
        # raise IOError( "Can not read object '%s' from file '%s'"%( name, filename ) )

    # return o

def read_object_auto( filename, name ):
    """Load an object from npz/hdf5/ROOT file"""

def make_iav( filename, name ):
    """Assembles a chain for IAV detector effect using input matrix from a file"""
    pass


