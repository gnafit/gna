#!/usr/bin/env python

"""Check Identity class"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import stdvector, Points
from gna.bindings import DataType

#
# Create the matrix
#
def test_points(opts):
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = Points(mat)
    identity = R.Identity()
    if opts.gpuargs:
        identity.identity.switchFunction('identity_gpuargs_h')

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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g', '--gpuargs', action='store_true')
    test_points(parser.parse_args())

