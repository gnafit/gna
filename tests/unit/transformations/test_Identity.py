#!/usr/bin/env python

"""Check Identity class"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna import constructors as C

def test_identity_01():
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = C.Points(mat)
    identity = C.Identity()
    # if opts.gpuargs:
        # identity.identity.switchFunction('identity_gpuargs_h')

    identity.identity.source( points.points.points )
    res = identity.identity.target.data()
    dt  = identity.identity.target.datatype()

    assert N.allclose(mat, res, atol=0, rtol=0), "C++ and Python results doesn't match"

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

def test_identity_02_fill():
    mat = N.arange(12, dtype='d').reshape(3, 4)
    mat1 = -mat

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    print( 'Second input matrix (numpy)' )
    print( mat1 )
    print()

    #
    # Create transformations
    #
    Points = C.Points(mat)
    points = Points.points
    Identity = C.Identity()
    identity = Identity.identity
    # if opts.gpuargs:
        # identity.identity.switchFunction('identity_gpuargs_h')

    points.points >> identity.source
    res = identity.target.data()
    dt  = identity.target.datatype()

    print( 'Result (C++ Data to numpy)' )
    print( res )
    print()

    assert N.allclose(mat, res, atol=0, rtol=0), "C++ and Python results doesn't match"

    slc1=mat1[:,0]
    filled = points.points.fill(slc1)

    assert filled==slc1.size
    assert not points.tainted()
    assert identity.tainted()

    res1 = identity.target.data()
    points1 = points.points.data()
    print('Result 1 (C++ Data to numpy)')
    print(res1)
    print()

    print('Points 1')
    print(points1)
    print()

    assert N.allclose(mat[:,1:], points1[:,1:], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat1[:,0], points1[:,0], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat[:,1:], res1[:,1:], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat1[:,0], res1[:,0], atol=0, rtol=0), "C++ and Python results doesn't match"

    slc2=2*mat[:,0]
    filled = identity.target.fill(slc2)

    assert filled==slc2.size
    assert not points.tainted()
    assert not identity.tainted()

    res2 = identity.target.data()
    points2 = points.points.data()
    print('Result 2 (C++ Data to numpy)')
    print(res2)
    print()

    print('Points 2')
    print(points2)
    print()

    assert N.allclose(mat[:,1:], points1[:,1:], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat1[:,0], points1[:,0], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat[:,1:], res2[:,1:], atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(2*mat[:,0], res2[:,0], atol=0, rtol=0), "C++ and Python results doesn't match"

def test_identity_03_verbose():
    mat = N.arange(12, dtype='d').reshape(3, 4)+1
    vec = mat[0]
    num = vec[:1]

    #
    # Create transformations
    #
    Mat = C.Points(mat)
    Vec = C.Points(vec)
    Num = C.Points(num)

    Identity = C.IdentityVerbose("test identity")
    Identity.add_input(Num.single())
    Identity.identity.touch()
    Identity.add_input(Vec.single())
    Identity.identity.touch()
    Identity.add_input(Mat.single())
    Identity.identity.touch()

    assert N.allclose(num, Identity.identity.outputs[0].data(), atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(vec, Identity.identity.outputs[1].data(), atol=0, rtol=0), "C++ and Python results doesn't match"
    assert N.allclose(mat, Identity.identity.outputs[2].data(), atol=0, rtol=0), "C++ and Python results doesn't match"
