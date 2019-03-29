#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the one integrator can provide binning for another through adapter class
"""

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import gna.constructors as C
from gna.bindings import DataType
from gna.env import env

def test_connect_integrators_with_adapter():
    initial_binning = np.arange(1, 10, 0.5)

    initial_integrator = C.IntegratorGL(initial_binning, 4, labels = (('First Sampler', 'First Integrator')))
    print("Integration points from first integral")
    print(initial_integrator.points.x)

    points_to_hist = C.PointsToHist(initial_integrator.points.x)
    print("Integration points converted to hist with filled left most edge as 0:")
    converted = np.concatenate((np.zeros(1), points_to_hist.adapter.hist.data()), axis=0)
    print(converted)

    second_integrator = R.IntegratorGL(len(points_to_hist.adapter.hist.data())-1, 4, labels = (('Second Sampler', 'Second Integrator')))
    second_integrator.points.edges(points_to_hist.adapter.hist)
    print("Integration points from second integrator")
    print(second_integrator.points.xedges.data())
    assert np.allclose(converted, second_integrator.points.xedges.data())

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
    


if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

