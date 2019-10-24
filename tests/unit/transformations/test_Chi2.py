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

def test_chi2_v01():
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')
    theorya = dataa+offset
    stata   = dataa**0.5

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')
    stat   = C.Points(stata, labels='Stat errors')

    chi = C.Chi2(labels='Chi2')
    chi.add(theory, data, stat)

    if graphviz:
        chi.print()
        from gna.graphviz import savegraph
        savegraph(data.single(), 'output/unit_chi2_v01.dot')

    res = chi.chi2.chi2.data()[0]
    res_expected1 = (((dataa-theorya)/stata)**2).sum()
    res_expected2 = ((offset/stata)**2).sum()

    assert (res==res_expected1).all()
    assert (res==res_expected2).all()

def test_chi2_v02():
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')
    theorya = dataa+offset
    covmat  = N.diag(dataa)
    La      = N.linalg.cholesky(covmat)

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')
    L      = C.Points(La, labels='Stat errors')

    chi = C.Chi2(labels='Chi2')
    chi.add(theory, data, L)

    if graphviz:
        chi.print()
        from gna.graphviz import savegraph
        savegraph(data.single(), 'output/unit_chi2_v02.dot')

    res = chi.chi2.chi2.data()[0]
    res_expected1 = (offset**2/dataa).sum()

    assert (res==res_expected1).all()

def test_chi2_v03():
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')
    theorya = dataa+offset
    covmat  = N.diag(dataa)+2.0
    La      = N.linalg.cholesky(covmat)

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')
    L      = C.Points(La, labels='Stat errors')

    chi = C.Chi2(labels='Chi2')
    chi.add(theory, data, L)

    if graphviz:
        chi.print()
        from gna.graphviz import savegraph
        savegraph(data.single(), 'output/unit_chi2_v03.dot')

    res = chi.chi2.chi2.data()[0]

    diff = N.array(dataa-theorya).T
    res_expected1 = N.matmul(diff.T, N.matmul(N.linalg.inv(covmat),  diff))
    ndiff = N.matmul(N.linalg.inv(La), diff)
    res_expected2 = N.matmul(ndiff.T, ndiff)

    assert N.allclose(res, res_expected1, rtol=0, atol=1.e-15)
    assert N.allclose(res, res_expected2, rtol=0, atol=1.e-15)

if __name__ == "__main__":
    run_unittests(globals())

