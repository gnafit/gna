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
from gna import context

def normalizedconvolution_prepare(fcn, weights, scale=1.0, nsname=None):
    fcn = N.ascontiguousarray(fcn, dtype=context.current_precision_short())
    weights = N.ascontiguousarray(weights, dtype=context.current_precision_short())

    fcn_p, weights_p = C.Points(fcn), C.Points(weights)

    if nsname is not None:
        ns = env.globalns(nsname)
        ns.defparameter('scale', central=scale, fixed=True)
        with ns:
            nc = C.NormalizedConvolution('scale')
    else:
        scale=1.0
        nc = C.NormalizedConvolution()

    fcn_p >> nc.normconvolution.fcn
    weights_p >> nc.normconvolution.weights

    res = nc.normconvolution.result.data()
    prod = nc.normconvolution.product.data()

    product_e = fcn*weights
    res_e = scale*product_e.sum()/weights.sum()

    print('Scale', scale)
    print('Function', fcn)
    print('Weights', weights)
    print('Product', prod)
    print('Product expected', product_e)
    print('Result', res)
    print('Result expected', res_e)

    assert (prod==product_e).all()
    assert (res==res_e).all()
    print()

@floatcopy(globals(), addname=True)
def test_normalizedconvolution_v01(function_name):
    normalizedconvolution_prepare([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    normalizedconvolution_prepare([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    normalizedconvolution_prepare([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    normalizedconvolution_prepare([3.0, 4.0, 5.0], [3.0, 2.0, 1.0])

    normalizedconvolution_prepare([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], scale=3.0, nsname=function_name)

if __name__ == "__main__":
    run_unittests(globals())

