#!/usr/bin/env python
import load
import ROOT
import numpy as np
from gna.env import env

def test_ParCenters():
    test_ns = env.ns("test_ParCenters")
    centers = np.arange(10)
    pars = [test_ns.defparameter(f"par{center:02d}", central=center, sigma=1) for center in centers]

    pars_centers = ROOT.ParCenters()
    for par in pars: pars_centers.append(par)
    pars_centers.materialize()

    res = pars_centers.centers.centers.data()
    assert np.allclose(res, centers, atol=0, rtol=0)

