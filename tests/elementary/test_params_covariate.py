#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
import numpy as np
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
from gna.config import cfg
import gna.parameters.covariance_helpers as ch
import itertools

def make_fake_covmat(dimension):
    eigen_decomp = np.eye(dimension)*0.5
    triued_mat = np.triu(np.ones(dimension))
    transform_mat = np.matrix(triued_mat/np.sum(triued_mat, axis=0))
    return np.matmul(transform_mat.T, np.matmul(eigen_decomp, transform_mat))

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject

def test_par_cov():
    probe_1 = env.defparameter('probe1', central=0., sigma=1.)
    probe_2 = env.defparameter('probe2', central=0., sigma=1.)
    probe_3 = env.defparameter('probe3', central=0., sigma=1.)
    test_ns = env.ns('test_ns')
    test_ns.defparameter('test0', central=1., sigma=0.1)
    test_ns.defparameter('test1', central=1., sigma=0.1)
    test_ns.defparameter('test2', central=1., sigma=0.1)
    test_ns.defparameter('test3', central=1., sigma=0.1)
    extra_test_ns = env.ns('extra_test_ns')
    extra1 = extra_test_ns.defparameter('extra1', central=1., sigma=0.1)
    extra2 = extra_test_ns.defparameter('extra2', central=1., sigma=0.1)
    extra3 = extra_test_ns.defparameter('extra3', central=1., sigma=0.1)
    extra4 = extra_test_ns.defparameter('extra4', central=1., sigma=0.1)

    cov1 = 0.1
    print("Setting covariance of probe_1 with probe_2 to {0}".format(cov1))
    probe_1.setCovariance(probe_2, cov1)
    print("Check that they are mutually correlated now.")

    assert probe_1.isCorrelated(probe_2) and probe_2.isCorrelated(probe_1)
    print("Success")
    print("Get covariance from both -- {0} and {1}\n".format(probe_1.getCovariance(probe_2), probe_2.getCovariance(probe_1)))

    print("Checks that change of one propagates to another")
    cov2 = 0.2
    probe_1.setCovariance(probe_2, cov2)
    assert (probe_1.getCovariance(probe_2) == cov2 and probe_2.getCovariance(probe_1) == cov2)
    print("Success\n")


    test_pars = get_parameters(['test_ns.test0', 'test_ns.test1', 'test_ns.test2', 'test_ns.test3'])
    print("Test pars sequence is {}".format([_.name() for _ in test_pars]))
    cov_matrix1 = make_fake_covmat(4)
    print("Test covariance matrix is \n", cov_matrix1)

    ch.covariate_pars(test_pars, cov_matrix1)
    for first, second in itertools.combinations_with_replacement(range(len(test_pars)), 2):
        try:
            if first != second:
                assert test_pars[first].getCovariance(test_pars[second]) == cov_matrix1[first, second]
            else:
                assert test_pars[first].sigma() == np.sqrt(cov_matrix1[first, second])
        except AssertionError:
            print((first, second),
                    test_pars[first].getCovariance(test_pars[second])**2,
                    cov_matrix1[first, second])
            raise

    extra_pars = [extra1, extra2, extra3]
    cov_mat_extra = make_fake_covmat(3)
    cov_storage = ch.CovarianceStorage("extra_store", extra_pars, cov_mat_extra)
    ch.covariate_ns('extra_test_ns', cov_storage)
    for first, second in itertools.combinations_with_replacement(range(len(extra_pars)), 2):
        try:
            if first != second:
                assert test_pars[first].getCovariance(test_pars[second]) == cov_matrix1[first, second]
            else:
                assert test_pars[first].sigma() == np.sqrt(cov_matrix1[first, second])
        except AssertionError:
            print((first, second),
                    test_pars[first].getCovariance(test_pars[second])**2,
                    cov_matrix1[first, second])
            raise

if __name__ == "__main__":
    test_par_cov()
