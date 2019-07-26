#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from gna import constructors as C
from gna.env import env
from gna.unittest import *
import numpy as N

graphviz = False

def test_covariated_predictionV_v01():
    n = 10
    start = 10
    dataa   = N.arange(start, start+n)

    data = C.Points(dataa, labels='Data')
    stat = C.Points(dataa, labels='Stat errors')

    cp = C.CovariatedPredictionV(labels=['Concatenated prediction', 'Concatenated base covariance matrix', 'Full cov mat Cholesky decomposition'])
    cp.append(data)
    cp.covariate(stat, data, 1, data, 1)
    cp.covbase.L >> cp.cov.Lbase
    cp.finalize()

    if graphviz:
        cp.print()
        from gna.graphviz import savegraph
        savegraph([data.single(),cp.covbase.L], 'output/unit_covariated_predictionv_01.dot')

    datao   = cp.prediction.prediction.data()
    covbase = cp.covbase.L.data()
    L       = cp.cov.L.data()
    L_expect = dataa**0.5

    assert (dataa==datao).all()
    assert (covbase==dataa).all()
    assert (L==L_expect).all()

if __name__ == "__main__":
    run_unittests(globals())

