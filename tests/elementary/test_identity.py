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
    __testing_points(is_gpu=False)

def __testing_points(is_gpu):
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = Points(mat)
    identity = R.Identity()
    if is_gpu:
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
    args = parser.parse_args()
    is_gpu = True if args.gpuargs else False
    __testing_points(is_gpu)

