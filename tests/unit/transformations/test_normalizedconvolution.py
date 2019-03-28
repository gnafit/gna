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

@floatcopy(globals())
def normalizedconvolution_prepare(fcn, weights):
    fcn = N.ascontiguousarray(fcn, dtype='d')
    weights = N.ascontiguousarray(weights, dtype='d')

    fcn_p, weights_p = C.Points(fcn), C.Points(weights)
    nc = C.NormalizedConvolution()

    fcn_p >> nc.normconvolution.fcn
    weights_p >> nc.normconvolution.weights

    res = nc.normconvolution.result.data()
    prod = nc.normconvolution.product.data()

    product_e = fcn*weights
    res_e = product_e.sum()/weights.sum()

    print('Function', fcn)
    print('Weights', weights)
    print('Product', prod)
    print('Product expected', product_e)
    print('Result', res)
    print('Result expected', res_e)

    assert (prod==product_e).all()
    assert (res==res_e).all()
    print()

def test_normalizedconvolution_v01():
    normalizedconvolution_prepare([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    normalizedconvolution_prepare([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    normalizedconvolution_prepare([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    normalizedconvolution_prepare([3.0, 4.0, 5.0], [3.0, 2.0, 1.0])

if __name__ == "__main__":
    run_unittests(globals())

