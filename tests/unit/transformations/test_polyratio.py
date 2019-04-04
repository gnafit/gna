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

def polyratio_prepare(nsname, nominator, denominator):
    inp = N.arange(1, 11, dtype='d')
    x = C.Points(inp)
    ns = env.globalns(nsname)

    nominator_names=[]
    denominator_names=[]

    nominator_exp=0.0 if nominator else 1.0
    cpower = 1.0
    for i, nom in enumerate(nominator):
        name = 'nom%i'%i
        ns.defparameter(name, central=nom, fixed=True)
        nominator_names.append(name)
        nominator_exp+=nom*cpower
        cpower*=inp

    denominator_exp=0.0 if denominator else 1.0
    cpower = 1.0
    for i, denom in enumerate(denominator):
        name = 'denom%i'%i
        ns.defparameter(name, central=denom, fixed=True)
        denominator_names.append(name)
        denominator_exp+=denom*cpower
        cpower*=inp

    with ns:
        pratio=C.PolyRatio(nominator_names, denominator_names)
    x >> pratio.polyratio.points

    res = pratio.polyratio.ratio.data()
    res_exp = nominator_exp/denominator_exp

    print('Nominator weights', nominator)
    print('Denominator weights', denominator)
    print('Result', res)
    print('Expected', res_exp)
    print()

    assert(N.allclose(res, res_exp))

def test_polyratio_v00():
    polyratio_prepare('test_polyratio_v01', [0.0, 1.0], [])
    polyratio_prepare('test_polyratio_v02', [0.0, 2.0], [])
    polyratio_prepare('test_polyratio_v03', [1.0, 0.0], [])
    polyratio_prepare('test_polyratio_v04', [2.0, 0.0], [])
    polyratio_prepare('test_polyratio_v05', [], [0.0, 1.0])
    polyratio_prepare('test_polyratio_v06', [], [0.0, 2.0])
    polyratio_prepare('test_polyratio_v07', [], [1.0, 0.0])
    polyratio_prepare('test_polyratio_v08', [], [2.0, 0.0])

    polyratio_prepare('test_polyratio_v09', [0.0, 0.0, 2.0], [])
    polyratio_prepare('test_polyratio_v10', [], [0.0, 0.0, 2.0])

    polyratio_prepare('test_polyratio_v11', [0.0, 2.0], [0.0, 1.0])
    polyratio_prepare('test_polyratio_v12', [0.0, 0.0, 2.0], [0.0, 2.0])
    polyratio_prepare('test_polyratio_v13', [0.0, 0.0, 2.0], [0.0, 0.0, 2.0])

if __name__ == "__main__":
    run_unittests(globals())

