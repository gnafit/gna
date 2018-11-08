#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
import numpy as np
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
from gna.config import cfg
import itertools

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject

def test_defpar_group():
    test_ns = env.ns("test")
    covmat = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])

    pars = test_ns.defparameter_group(('par1', {'central': 1.0, 'relsigma': 0.1}),
                                      ('par2', {'central': 2.0, 'relsigma': 0.1}),
                                      ('par3', {'central': 3.0, 'relsigma': 0.1}),
                                      **{'covmat': covmat, })
    for first, second in itertools.combinations_with_replacement(range(len(pars)), 2):
        cov_from_pars = pars[first].getCovariance(pars[second])
        cov_direct = pars[first].sigma() * pars[second].sigma() * covmat[first, second]
        assert np.allclose(cov_from_pars, cov_direct), "Covs doesn't match!"

if __name__ == "__main__":
    test_defpar_group()
