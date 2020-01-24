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
from gna.unittest import allure_attach_file, savegraph
from scipy.special import gammaln
import pytest

graphviz = False

@pytest.mark.parametrize('cls', [R.Poisson, R.LnPoissonSplit])
@pytest.mark.parametrize('approx', [True, False])
def test_lnpoisson_v01(cls, approx, tmp_path):
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')
    theorya = dataa+offset

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')

    lnpoisson = cls(approx, labels='LnPoisson')
    lnpoisson.add(theory, data)

    if graphviz:
        lnpoisson.print()
        from gna.graphviz import savegraph
        savegraph(data.single(), 'output/unit_lnpoisson_v01.dot')

    res = lnpoisson.poisson.poisson.data()[0]

    if approx:
        res_expected = 2.0*((theorya - dataa*N.log(theorya)).sum() + (dataa*N.log(dataa)).sum())
    else:
        res_expected = 2.0*((theorya - dataa*N.log(theorya)).sum() + gammaln(dataa+1).sum())

    assert N.isclose(res, res_expected)

if __name__ == "__main__":
    run_unittests(globals())

