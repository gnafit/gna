#!/usr/bin/env python
import load
import ROOT
import numpy as np
from gna.env import env
import pytest

@pytest.mark.parametrize('mode', ['full2d', 'diag2d', 'diag1d'])
def test_ParCovMatrix(mode: str):
    test_ns = env.ns(f"test_ParCovMatrix_{mode}")
    relsigma=0.1
    p1 = test_ns.defparameter("par1", central=1.0, relsigma=relsigma)
    p2 = test_ns.defparameter("par2", central=2.0, relsigma=relsigma)
    p3 = test_ns.defparameter("par3", central=3.0, relsigma=relsigma)
    pars = (p1, p2, p3)

    # Not real covariance matrix. Just to check that covariances propagates
    # correctly into composite parameter covariance matrix

    if mode=='full2d':
        pars_covmat = ROOT.ParCovMatrix()
        p1.setCovariance(p2, 0.1)
        p1.setCovariance(p3, 0.2)
        p2.setCovariance(p3, 0.5)
        python_covmat = np.array([[(1*relsigma)**2, 0.1,             0.2],
                                   [0.1,            (2*relsigma)**2, 0.5],
                                   [0.2,            0.5,             (3*relsigma)**2]])
    elif mode=='diag2d':
        pars_covmat = ROOT.ParCovMatrix()
        python_covmat = np.array([[(1*relsigma)**2, 0.0,             0.0],
                                   [0.0,            (2*relsigma)**2, 0.0],
                                   [0.0,            0.0,             (3*relsigma)**2]])
    else:
        pars_covmat = ROOT.ParCovMatrix(ROOT.GNA.MatrixFormat.PermitDiagonal)
        python_covmat = np.array([(1*relsigma)**2, (2*relsigma)**2, (3*relsigma)**2])

    for p in pars:
        pars_covmat.append(p)
    pars_covmat.materialize()

    pars_covmat.printtransformations()

    print('Python:')
    print(python_covmat)
    print('Transformation:')
    print(pars_covmat.unc_matrix.data())
    msg = "Covmatrices from python and C++ doesn't match"
    assert np.allclose(python_covmat, pars_covmat.unc_matrix.data(), rtol=0, atol=0), msg
