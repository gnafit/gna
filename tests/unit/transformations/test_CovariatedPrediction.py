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

def test_covariated_prediction_v01():
    n = 10
    start = 10
    dataa   = N.arange(start, start+n)

    data = C.Points(dataa, labels='Data')
    stat = C.Points(dataa, labels='Stat errors')

    cp = C.CovariatedPrediction(labels=['Concatenated prediction', 'Concatenated base covariance matrix', 'Full cov mat Cholesky decomposition'])
    cp.append(data)
    cp.covariate(stat, data, 1, data, 1)
    cp.covbase.L >> cp.cov.Lbase
    cp.finalize()


    if graphviz:
        cp.print()
        from gna.graphviz import savegraph
        savegraph([data.single(),cp.covbase.L], 'output/unit_covariated_prediction_01.dot')

    datao   = cp.prediction.prediction.data()
    covbase = cp.covbase.L.data()
    L = cp.cov.L.data()

    covbase_expect = N.diag(dataa)
    L_expect = N.linalg.cholesky(covbase_expect)

    assert (dataa==datao).all()
    assert (covbase==covbase_expect).all()
    assert (L==L_expect).all()

def test_covariated_prediction_v02():
    n = 10
    start = 10
    dataa   = N.arange(start, start+n)
    systa = N.ones((n,n), dtype='d')*1.5

    data = C.Points(dataa, labels='Data')
    stat = C.Points(dataa, labels='Stat errors')
    syst = C.Points(systa, labels='Syst matrix')

    cp = C.CovariatedPrediction(labels=['Concatenated prediction', 'Concatenated base covariance matrix', 'Full cov mat Cholesky decomposition'])
    cp.append(data)
    cp.covariate(stat, data, 1, data, 1)
    cp.addSystematicCovMatrix(syst)
    cp.covbase.L >> cp.cov.Lbase
    cp.finalize()

    if graphviz:
        cp.print()
        from gna.graphviz import savegraph
        savegraph([data.single(),cp.covbase.L], 'output/unit_covariated_prediction_02.dot')

    datao   = cp.prediction.prediction.data()
    covbase = cp.covbase.L.data()
    L = N.tril(cp.cov.L.data())

    covbase_expect = N.diag(dataa)
    L_expect = N.linalg.cholesky(covbase_expect+systa)

    assert (dataa==datao).all()
    assert (covbase==covbase_expect).all()
    assert N.allclose(L, L_expect)

if __name__ == "__main__":
    run_unittests(globals())

