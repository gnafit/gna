#!/usr/bin/env python

"""Check the WeightedSum transformation"""

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

@pytest.mark.parametrize('cls', [R.Poisson, R.LogPoissonSplit])
@pytest.mark.parametrize('approx', [True, False])
def test_logpoisson_v01(cls, approx, tmp_path):
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')
    theorya = dataa+offset

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')

    logpoisson = cls(approx, labels='LogPoisson')
    logpoisson.add(theory, data)

    res = logpoisson.poisson.poisson.data()[0]

    if approx:
        res_expected = 2.0*((theorya - dataa) + dataa*N.log(dataa/theorya)).sum()
    else:
        res_expected = 2.0*((theorya - dataa*N.log(theorya)).sum() + gammaln(dataa+1).sum())

    assert N.allclose(res, res_expected, atol=1.e-13, rtol=0)

