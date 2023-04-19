#!/usr/bin/env python

import numpy as np
from load import ROOT as R
from gna import constructors as C

def test_tf1():
    x = np.linspace(-10.0, 10.0, 401)
    def fcnpy(x, pars=None):
        return x[0]*x[0]
    fcnr = R.TF1("test1", fcnpy, x[0], x[-1])

    X = C.Points(x)
    O = C.TransformationTF1(fcnr, X.single())

    # O.printtransformations()

    y = fcnpy([x])
    ret = O.tf1.result()

    assert (ret==y).all()

def test_tf2():
    x = np.linspace(-10.0, 10.0, 201)
    y = np.linspace(-9.0, 9.0, 181)
    def fcnpy(x, pars=None):
        return x[0]*x[0] - x[1]*x[1]
    fcnr = R.TF2("test2", fcnpy, x[0], x[-1], y[0], y[-1])

    # fcnr.Draw('surf')
    # input()

    xx, yy = np.meshgrid(x, y, indexing='ij')

    X = C.Points(xx)
    Y = C.Points(yy)
    O = C.TransformationTF2(fcnr, X.single(), Y.single())

    # O.printtransformations()

    y = fcnpy([xx, yy])
    ret = O.tf2.result()

    assert (ret==y).all()
