#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from converters import convert

ns = env.globalns

raw_data = np.arange(100, dtype='d')
data = convert(raw_data, 'stdvector' )
points = R.Points(data)
vp = R.ArraySum()
vp.sum.arr(points)
print(vp.sum.accumulated.data(), "Got after reduction in C++")
print(raw_data.sum(), "Expected result with Python")
assert np.allclose(vp.sum.accumulated.data(), raw_data.sum()), "The sums from C++ and Python doesn't match"
print("Success!")



