#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Identity class"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import stdvector, Points
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
    points = Points(mat)
    identity = R.Identity()
    identity.identity.switchFunction('identity_gpu')

    identity.identity.source( points.points.points )
    res = identity.identity.target.data()
    dt  = identity.identity.target.datatype()

    assert N.allclose(mat, res), "C++ and Python results doesn't match"

    #
    # Dump
    #
    print( 'Eigen dump (C++)' )
    identity.dump()
    print()

    print( 'Result (C++ Data to numpy)' )
    print( res )
    print()

    print( 'Datatype:', str(dt) )


if __name__ == "__main__":
    test_points()

