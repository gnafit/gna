#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
from converters import convert
import numpy as np
from gna.env import env

test_ns = env.ns("test")
p1 = test_ns.defparameter("par1", central=1.0, relsigma=0.1)
p2 = test_ns.defparameter("par2", central=2.0, relsigma=0.1)
p3 = test_ns.defparameter("par3", central=3.0, relsigma=0.1)

# Not real covariance matrix. Just to check that covariances propagates
# correctly into composite parameter covariance matrix
p1.setCovariance(p2, 0.1)
p1.setCovariance(p3, 0.2)
p2.setCovariance(p3, 0.5)
python_covmat = np.matrix([[(1*0.1)**2, 0.1, 0.2],
                           [0.1, (2*0.1)**2, 0.5],
                           [0.2, 0.5, (3*0.1)**2]])

pars_covmat = ROOT.ParCovMatrix()
pars_covmat.append(p1)
pars_covmat.append(p2)
pars_covmat.append(p3)
pars_covmat.materialize()

print(pars_covmat.unc_matrix.data())
msg = "Covmatrices from python and C++ doesn't match"
assert np.allclose(python_covmat, pars_covmat.unc_matrix.data()), msg
