#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check Identity class"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import stdvector, Points, Dummy
from gna.bindings import DataType

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
    points = Points(mat)
    identity = R.Identity()
    if opts.function=='host':
        identity.identity.switchFunction('identity_gpuargs_h')
    else:
        identity.identity.switchFunction('identity_gpuargs_d')

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

def test_vars(opts):
    print('Test inputs/outputs/variables (Dummy)')
    mat1 = N.arange(12, dtype='d').reshape(3, 4)
    mat2 = N.arange(15, dtype='d').reshape(5, 3)

    from gna.env import env
    ns = env.globalns
    ns.reqparameter('par1', central=1.0,    fixed=True, label='Dummy parameter 1')
    ns.reqparameter('par2', central=1.5,    fixed=True, label='Dummy parameter 2')
    ns.reqparameter('par3', central=1.01e5, fixed=True, label='Dummy parameter 3')
    ns.printparameters(labels=True)

    points1, points2 = Points(mat1), Points(mat2)
    dummy = Dummy(4, "dummy", ['par1', 'par2', 'par3'])
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('-g', '--gpuargs', action='store_true')
    parser.add_argument('function', default='host', nargs='?',
                                    choices=['host', 'device'])

    test_io(parser.parse_args())
    #test_vars(parser.parse_args())
