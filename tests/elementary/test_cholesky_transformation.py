#!/usr/bin/env python

"""Check the Cholesky transformation"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import Points

#
# Create the matrix in numpy
#
def test_chol():
    size = 4
    v = N.matrix(N.arange(size, dtype='d'))
    v.A1[size//2:] = N.arange(size//2, 0, -1)

    mat = v.T*v + N.eye( size, size )*size*2

    chol = N.linalg.cholesky( mat )

    print( 'Matrix (numpy)' )
    print( mat )
    print()

    print( 'L (numpy)' )
    print( chol )
    print()

    #
    # Create the transformations
    #
    points = Points( mat )
    cholesky = R.Cholesky()

    cholesky.cholesky.mat( points.points.points )

    #
    # Retrieve data
    #
    res = cholesky.cholesky.L.data()
    res = N.matrix(N.tril( res ))

    #
    # Print data
    #
    print( 'L' )
    print( res )

    assert N.allclose(chol, res), "C++ Cholesky and Python one doesn't match"

    mat_back = res*res.T

    print( 'Matrix (rec)' )
    print( mat_back )
    assert N.allclose(mat, mat_back), "C++ result and Python origin doesn't match"

    diff = chol - res
    print( 'Diff L' )
    print( diff )

    diff1 = mat_back - mat
    print( 'Diff mat' )
    print( diff1 )

    print( (((N.fabs(diff)+N.fabs(diff1))>1.e-12).sum() and '\033[31mFail!' or '\033[32mOK!' ), '\033[0m' )

if __name__ == "__main__":
    test_chol()
