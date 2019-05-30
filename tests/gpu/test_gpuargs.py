#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Identity class"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna import constructors as C
from gna.bindings import DataType
from gna.unittest import *
from gna import context

#
# Create the matrix
#
def test_io(opts):
    print('Test inputs/outputs (Identity)')
    mat = N.arange(12, dtype='d').reshape(3, 4)

    print( 'Input matrix (numpy)' )
    print( mat )
    print()

    #
    # Create transformations
    #
    points = C.Points(mat)
    identity = C.Identity()
    identity.identity.switchFunction('identity_gpuargs_h')

    points.points.points >> identity.identity.source

    identity.print()

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

def gpuargs_make(nsname, mat1, mat2):
    from gna.env import env
    ns = env.globalns(nsname)
    ns.reqparameter('par1', central=1.0,    fixed=True, label='Dummy parameter 1')
    ns.reqparameter('par2', central=1.5,    fixed=True, label='Dummy parameter 2')
    ns.reqparameter('par3', central=1.01e5, fixed=True, label='Dummy parameter 3')
    ns.printparameters(labels=True)

    points1, points2 = C.Points(mat1), C.Points(mat2)
    with ns:
        dummy = C.Dummy(4, "dummy", ['par1', 'par2', 'par3'])

    return dummy, points1, points2, ns

@floatcopy(globals(), addname=True)
def test_vars_01_local(opts, function_name):
    print('Test inputs/outputs/variables (Dummy)')
    mat1 = N.arange(12, dtype='d').reshape(3, 4)
    mat2 = N.arange(15, dtype='d').reshape(5, 3)

    dummy, points1, points2, ns = gpuargs_make(function_name, mat1, mat2)

    dummy.dummy.switchFunction('dummy_gpuargs_h_local')
    dummy.add_input(points1, 'input1')
    dummy.add_input(points2, 'input2')
    dummy.add_output('out1')
    dummy.add_output('out2')

    dummy.print()

    res1 = dummy.dummy.out1.data()
    res2 = dummy.dummy.out2.data()
    dt1  = dummy.dummy.out1.datatype()
    dt2  = dummy.dummy.out2.datatype()

    assert N.allclose(res1, 0.0), "C++ and Python results doesn't match"
    assert N.allclose(res2, 1.0), "C++ and Python results doesn't match"

    print( 'Result (C++ Data to numpy)' )
    print( res1 )
    print( res2 )
    print()

    print( 'Datatype:', str(dt1) )
    print( 'Datatype:', str(dt2) )

    print('Change 3d variable')
    ns['par3'].set(-1.0)
    res1 = dummy.dummy.out1.data()

@floatcopy(globals(), addname=True)
def test_vars_02_host(opts, function_name):
    print('Test inputs/outputs/variables (Dummy)')
    mat1 = N.arange(12, dtype='d').reshape(3, 4)
    mat2 = N.arange(15, dtype='d').reshape(5, 3)

    with context.manager(100) as manager:
        dummy, points1, points2, ns = gpuargs_make(function_name, mat1, mat2)
        manager.setVariables(C.stdvector([par.getVariable() for (name, par) in ns.walknames()]))

    dummy.dummy.switchFunction('dummy_gpuargs_h')
    dummy.add_input(points1, 'input1')
    dummy.add_input(points2, 'input2')
    dummy.add_output('out1')
    dummy.add_output('out2')

    dummy.print()

    res1 = dummy.dummy.out1.data()
    res2 = dummy.dummy.out2.data()
    dt1  = dummy.dummy.out1.datatype()
    dt2  = dummy.dummy.out2.datatype()

    assert N.allclose(res1, 0.0), "C++ and Python results doesn't match"
    assert N.allclose(res2, 1.0), "C++ and Python results doesn't match"

    print( 'Result (C++ Data to numpy)' )
    print( res1 )
    print( res2 )
    print()

    print( 'Datatype:', str(dt1) )
    print( 'Datatype:', str(dt2) )

    print('Change 3d variable')
    ns['par3'].set(-1.0)
    res1 = dummy.dummy.out1.data()

@floatcopy(globals(), addname=True)
def test_vars_02_dev(opts, function_name):
    print('Test inputs/outputs/variables (Dummy)')
    mat1 = N.arange(12, dtype='d').reshape(3, 4)
    mat2 = N.arange(15, dtype='d').reshape(5, 3)

    with context.manager(100) as manager:
        dummy, points1, points2, ns = gpuargs_make(function_name, mat1, mat2)
        manager.setVariables(C.stdvector([par.getVariable() for (name, par) in ns.walknames()]))

    dummy.dummy.switchFunction('dummy_gpuargs_d')
    dummy.add_input(points1, 'input1')
    dummy.add_input(points2, 'input2')
    dummy.add_output('out1')
    dummy.add_output('out2')

    dummy.print()

    res1 = dummy.dummy.out1.data()
    res2 = dummy.dummy.out2.data()
    dt1  = dummy.dummy.out1.datatype()
    dt2  = dummy.dummy.out2.datatype()

    assert N.allclose(res1, 0.0), "C++ and Python results doesn't match"
    assert N.allclose(res2, 1.0), "C++ and Python results doesn't match"

    print( 'Result (C++ Data to numpy)' )
    print( res1 )
    print( res2 )
    print()

    print( 'Datatype:', str(dt1) )
    print( 'Datatype:', str(dt2) )

    print('Change 3d variable')
    ns['par3'].set(-1.0)
    res1 = dummy.dummy.out1.data()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g', '--gpuargs', action='store_true')

    run_unittests(globals(), parser.parse_args())

