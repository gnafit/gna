#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the Ratio transformation"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from constructors import Points

def test_ratio():
    scale = 2.0
    num   = 9
    step  = 2
    top    = N.arange(-num, num, step)
    bottom = top*scale

    top_o = Points(top)
    bottom_o = Points(bottom)

    ratio_o = R.Ratio(top_o, bottom_o)
    result = ratio_o.ratio.ratio.data()

    print('Scale', 1.0/scale)
    print('Top', top)
    print('Bottom', bottom)
    print('Result (python)', top/bottom)
    print('Result (GNA)', result)

    check = N.allclose(result, 1.0/scale)
    assert check, "Not matching results" 
    print( check and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

if __name__ == "__main__":
    test_ratio()
