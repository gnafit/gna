#!/usr/bin/env python

"""Check the WeightedSum transformation"""

import numpy as N
from load import ROOT as R
from gna import constructors as C
import numpy as N

def test_Chi2CNPStat_v01():
    n = 10
    start = 10
    offset = 1.0
    dataa   = N.arange(start, start+n, dtype='d')+1
    theorya = dataa+offset

    data   = C.Points(dataa, labels='Data')
    theory = C.Points(theorya, labels='Theory')

    chi2 = C.Chi2CNPStat(labels='chi2')
    chi2.add(theory, data)

    # chi2.printtransformations()
    res = chi2.chi2.chi2.data()[0]

    res_expected = ((1.0/dataa + 2.0/theorya)*(theorya - dataa)**2).sum()/3.0

    assert N.allclose(res, res_expected, rtol=1.e-13, atol=0)

