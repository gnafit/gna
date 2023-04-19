#!/usr/bin/env python

"""Check the Cholesky transformation"""

import numpy as np
from gna import constructors as C
import pytest

@pytest.mark.parametrize('ndim', [2, 1])
def test_chol(ndim: int) -> None:
    size = 12

    v = np.arange(size, dtype='d').reshape(1,size)
    v[0,size//2:] = np.arange(size//2, 0, -1)

    if ndim==2:
        mat = v.T@v + np.eye(size, size)*size*2
        chol = np.linalg.cholesky( mat )
    else:
        mat = v[0]
        chol = mat**0.5

    print( 'Matrix (numpy)' )
    print( mat )
    print()

    print( 'L (numpy)' )
    print( chol )
    print()

    #
    # Create the transformations
    #
    points = C.Points(mat)
    cholesky = C.Cholesky(int(ndim==1))

    points.points.points >> cholesky.cholesky.mat

    #
    # Retrieve data
    #
    res = cholesky.cholesky.L.data()
    if ndim==2:
        res = np.tril( res )

    #
    # Print data
    #
    print( 'L' )
    print( res )

    assert np.allclose(chol, res, rtol=0, atol=1.e-14), "C++ Cholesky and Python one doesn't match"

    if ndim==2:
        mat_back = res@res.T
    else:
        mat_back = res*res.T

    print( 'Matrix (rec)' )
    print( mat_back )
    assert np.allclose(mat, mat_back, rtol=0, atol=1.e-14), "C++ result and Python origin doesn't match"

    # diff = chol - res
    # print( 'Diff L' )
    # print( diff )
    #
    # diff1 = mat_back - mat
    # print( 'Diff mat' )
    # print( diff1 )
    #
    # print( (((np.fabs(diff)+np.fabs(diff1))>1.e-12).sum() and '\033[31mFail!' or '\033[32mOK!' ), '\033[0m' )
