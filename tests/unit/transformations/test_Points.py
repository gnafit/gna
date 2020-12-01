#!/usr/bin/env python

"""Check Points class

and test the matrix memory ordering"""

from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as N
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna import context
from gna.bindings import DataType, provided_precisions
from gna.unittest import *

@floatcopy(globals())
def test_points_v01():
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = C.Points(mat)
    identity = C.Identity()

    identity.identity.source( points.points.points )
    res = identity.identity.target.data()
    dt  = identity.identity.target.datatype()

    #
    # Dump
    #
    print( 'Eigen dump (C++)' )
    identity.dump()
    print()

    print( 'Points output' )
    print(points.points.points.data())

    print( 'Result (C++ Data to numpy)' )
    print( res, res.dtype )
    print()

    print( 'Datatype:', str(dt) )

    assert N.allclose(mat, res), "C++ and Python results doesn't match"

@floatcopy(globals())
def test_points_v02():
    arr = N.arange(12, dtype=context.current_precision_short())

    points = C.Points(arr)
    identity = C.Identity()
    points >> identity

    out = identity.single()

    for i in range(arr.size):
        points.set(arr, i)
        assert (out.data()==arr[:i]).all()

if __name__ == "__main__":
    run_unittests(globals())

