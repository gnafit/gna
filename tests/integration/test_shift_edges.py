#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check different ways of shifting bin edges
"""
from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.env import env

def test_shifting_edges_1():
    par = env.defparameter("par", central=0.1, fixed=True)
    dummy = env.defparameter("dummy", central=1, fixed=True)
    
    initial_binning = np.arange(1, 10, 0.5)

    initial_integrator = C.IntegratorGL(initial_binning, 4, labels = (('First Sampler', 'First Integrator')))
    print("Integration edges from first integral")
    print(initial_integrator.points.xedges.data())

    filled = C.FillLike(-2)
    initial_integrator.points.xedges >> filled.fill
    outputs = [par.single(), filled.single(), dummy.single(), initial_integrator.points.xedges]
    shifted = C.WeightedSumP(outputs)
    print("After shift by -2*{}".format(par.value()))
    print(shifted.sum.sum.data())
    expected = initial_binning - 2*par.value()
    assert np.allclose(expected, shifted.sum.sum.data())

def test_shifting_edges_2():
    param = env.defparameter("param", central=0.1, fixed=True)
    placeholder = env.defparameter("placeholder", central=1, fixed=True)
    
    initial_binning = np.arange(1, 10, 0.5)

    initial_integrator = C.IntegratorGL(initial_binning, 4, labels = (('First Sampler', 'First Integrator')))
    print("Integration edges from first integral")
    print(initial_integrator.points.xedges.data())

    from gna.constructors import stdvector
    shifted = R.WeightedSum(-2*param.value(), stdvector(['placeholder']), stdvector(['inp']))
    shifted.sum.inp(initial_integrator.points.xedges)
    print("After shift by -2*{}".format(param.value()))
    print(shifted.sum.sum.data())
    expected = initial_binning - 2*param.value()
    assert np.allclose(expected, shifted.sum.sum.data())

def test_shifting_edges_3():
    param = env.defparameter("param1", central=0.1, fixed=True)
    placeholder = env.defparameter("placeholder1", central=1, fixed=True)
    
    initial_binning = np.arange(1, 10, 0.5)

    initial_integrator = C.IntegratorGL(initial_binning, 4, labels = (('First Sampler', 'First Integrator')))
    print("Integration edges from first integral")
    print(initial_integrator.points.xedges.data())

    param_point = C.Points([-2*param.value()])
    inputs = [param_point.points.points, initial_integrator.points.xedges]
               
    shifted = C.SumBroadcast(inputs)

    print("After shift by -2*{}".format(param.value()))
    print(shifted.single().data())
    expected = initial_binning - 2*param.value()
    assert np.allclose(expected, shifted.single().data())

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

