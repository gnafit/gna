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

test_ns = env.ns("test")
covmat = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])

pars = test_ns.reqparameter_group(('par1', {'central': 1.0, 'relsigma': 0.1}),
                                  ('par2', {'central': 2.0, 'relsigma': 0.1}),
                                  ('par3', {'central': 3.0, 'relsigma': 0.1}),
                                  **{'covmat': covmat, })
for first, second in itertools.combinations_with_replacement(range(len(pars)), 2):
    cov_from_pars = pars[first].getCovariance(pars[second])
    cov_direct = pars[first].sigma() * pars[second].sigma() * covmat[first, second]
    assert np.allclose(cov_from_pars, cov_direct), "Covs doesn't match!"

extra_pars = test_ns.reqparameter_group(('extra1', {'central': 1.0, 'relsigma': 0.1}),
                                  ('extra2', {'central': 2.0, 'relsigma': 0.1}),
                                  ('extra3', {'central': 3.0, 'relsigma': 0.1}),
                                  **{'covmat_cfg': 'test_cov', })

covmat_cfg = cfg['covariances']['test_cov']['cov_mat']
try:
    covmat_cfg[0,0]
except TypeError:
    covmat_cfg = np.array(covmat_cfg)

for first, second in itertools.combinations_with_replacement(range(len(extra_pars)), 2):
    extra_from_pars = extra_pars[first].getCovariance(extra_pars[second])
    extra_direct = extra_pars[first].sigma() * extra_pars[second].sigma() * covmat_cfg[first, second]
    assert np.allclose(extra_from_pars, extra_direct), "Covs doesn't match!"


#Let's check for validity that no covariations if the parameters is already
#  present

new_ns = env.ns('new')
new_ns.defparameter("test1", central=1., relsigma=0.1)
new_pars = new_ns.reqparameter_group(('test1', {'central': 1.0, 'relsigma': 0.1}),
                                  ('test2', {'central': 2.0, 'relsigma': 0.1}),
                                  ('test3', {'central': 3.0, 'relsigma': 0.1}),
                                  **{'covmat': covmat, })
for first, second in itertools.combinations_with_replacement(range(len(new_pars)), 2):
    if first != second:
        assert not new_pars[first].isCovariated(new_pars[second]), "Covariance should not be assigned to pars!"


