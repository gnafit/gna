#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the Ratio transformation"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from gna.constructors import Points

def test_ratio():
    scale = 2.0
    num   = 9
    step  = 2
    top    = N.arange(-num, num, step)
    bottom = top*scale

    top_o = Points(top)
    bottom_o = Points(bottom)

    ratio_o = R.Ratio(top_o, bottom_o)
    out = ratio_o.ratio.ratio
    result = out.data()

    print('Scale', 1.0/scale)
    print('Top', top)
    print('Bottom', bottom)
    print('Result (python)', top/bottom)
    print('Result (GNA)', result)

    check = N.allclose(result, 1.0/scale, rtol=0, atol=1.e-15)
    assert check, "Not matching results"

    tf_top = top_o.points.getTaintflag()
    tf_bottom = bottom_o.points.getTaintflag()
    tf_r = ratio_o.ratio.getTaintflag()

    out.data()
    assert not tf_top.tainted() \
           and not tf_bottom.tainted() \
           and not tf_r.tainted()

    tf_top.taint()
    assert tf_top.tainted() \
           and not tf_bottom.tainted() \
           and tf_r.tainted()
    out.data()

    tf_bottom.taint()
    assert not tf_top.tainted() \
           and tf_bottom.tainted() \
           and tf_r.tainted()
    out.data()
