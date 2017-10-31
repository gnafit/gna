#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
import numpy as np
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
from gna.parameters.covariance_helpers import covariate_pars
import itertools

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject
print() 

probe_1 = env.defparameter('probe1', central=0., sigma=1.)
probe_2 = env.defparameter('probe2', central=0., sigma=1.)
probe_3 = env.defparameter('probe3', central=0., sigma=1.)
test_ns = env.ns('test_ns')
test_ns.defparameter('test0', central=1., sigma=0.1)
test_ns.defparameter('test1', central=1., sigma=0.1)
test_ns.defparameter('test2', central=1., sigma=0.1)
test_ns.defparameter('test3', central=1., sigma=0.1)
extra_test_ns = env.ns('extra_test_ns')
extra_test_ns.defparameter('extra1', central=1., sigma=0.1)
extra_test_ns.defparameter('extra2', central=1., sigma=0.1)
extra_test_ns.defparameter('extra3', central=1., sigma=0.1)
extra_test_ns.defparameter('extra4', central=1., sigma=0.1)

cov1 = 0.1
print("Setting covariance of probe_1 with probe_2 to {0}".format(cov1))
probe_1.setCovariance(probe_2, cov1)
print("Check that they are mutually covariated now.")
assert probe_1.isCovariated(probe_2) and probe_2.isCovariated(probe_1)
print("Success")
print("Get covariance from both -- {0} and {1}\n".format(probe_1.getCovariance(probe_2), probe_2.getCovariance(probe_1)))

print("Checks that change of one propagates to another")
cov2 = 0.2
probe_1.setCovariance(probe_2, cov2)
assert (probe_1.getCovariance(probe_2) == cov2 
        and probe_2.getCovariance(probe_1) == cov2)
print("Success\n")


test_pars = get_parameters(['test_ns.test0', 'test_ns.test1', 'test_ns.test2', 'test_ns.test3'])
print("Test pars sequence is {}".format([_.name() for _ in test_pars]))
cov_matrix1 = np.arange(4)*np.arange(4)[:, np.newaxis]
print("Test covariance matrix is \n", cov_matrix1)

covariate_pars(pars=test_pars, cov_matrix=cov_matrix1)
for first, second in itertools.combinations_with_replacement(range(len(test_pars)), 2):
    assert test_pars[first].getCovariance(test_pars[second]) == cov_matrix1[first, second]

