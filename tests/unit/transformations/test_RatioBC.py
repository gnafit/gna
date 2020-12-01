#!/usr/bin/env python

"""Check the Ratio transformation"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from gna.constructors import Points
import pytest

@pytest.mark.parametrize('mode', [0, 1, 2])
def test_ratio_bc(mode):
    if mode==0:
        res    = N.arange(1.0, 13.0).reshape(3,4)
        bottom = N.arange(1, 4)
        top    = res*bottom[:,None]
    elif mode==1:
        res    = N.arange(1.0, 13.0).reshape(3,4)
        top    = N.arange(1, 4)
        bottom = top[:,None]/res
    else:
        top     = N.arange(1.0, 13.0).reshape(3,4)*2
        bottom  = N.arange(1.0, 13.0).reshape(3,4)
        res     = top/bottom

    top_o = Points(top)
    bottom_o = Points(bottom)

    ratio_o = R.RatioBC(top_o, bottom_o)
    out = ratio_o.ratio.ratio
    result = out.data()

    print('Top', top)
    print('Bottom', bottom)
    print('Result (python)', res)
    print('Result (GNA)', result)

    check = N.allclose(result, res, rtol=0, atol=1.e-15)
    assert check, "Not matching results"

