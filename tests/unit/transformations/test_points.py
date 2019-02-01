#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Points class

and test the matrix memory ordering"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType

#
# Create the matrix
#
def test_points():
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = C.Points(mat)
    identity = R.Identity()

    identity.identity.source( points.points.points )
    res = identity.identity.target.data()
    dt  = identity.identity.target.datatype()

    #
    # Dump
    #
    print( 'Eigen dump (C++)' )
    identity.dump()
    print()

    print( 'Result (C++ Data to numpy)' )
    print( res, res.dtype )
    print()

    print( 'Datatype:', str(dt) )

    assert N.allclose(mat, res), "C++ and Python results doesn't match"


def test_pointsf():
    mat = N.arange(12, dtype='f').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    with C.precision('float'):
        points = C.Points(mat)

    out = points.points.points
    res = out.data()
    dt  = out.datatype()

    print( 'Result (C++ Data to numpy)' )
    print( res, res.dtype )
    print()

    print( 'Datatype:', str(dt) )

    assert N.allclose(mat, res), "C++ and Python results doesn't match"

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

