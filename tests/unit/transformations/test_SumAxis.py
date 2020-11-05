#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import gna.constructors as C
import pytest

@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['points', 'hist'])
def test_sumaxis_01(kind, axis):
    """Test ViewRear on Points (start, len)"""
    size=4
    inp = np.arange(12.0).reshape(3,4)
    shouldbe = inp.sum(axis=axis)

    xedges = np.arange(inp.shape[0]+1)
    yedges = np.arange(inp.shape[1]+1)
    edges = (xedges, yedges)
    if kind=='points':
        Inp = C.Points(inp)
    else:
        Inp = C.Histogram2d(xedges, yedges, inp)

    sum = C.SumAxis(axis, Inp)
    sum.printtransformations()

    res = sum.sumaxis.result.data()
    print('Input', inp)
    print('Result ({})'.format(axis), res)
    print('Should be', shouldbe)

    assert np.allclose(res, shouldbe, atol=0, rtol=0)
    if kind=='hist':
        newedges = sum.sumaxis.result.datatype().edges
        select = 1 if axis==0 else 0
        print('Original edges', edges)
        print('New edges', newedges)
        assert np.allclose(edges[select], newedges)


