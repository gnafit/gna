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


if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print('All tests are OK!')

