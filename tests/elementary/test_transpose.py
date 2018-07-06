#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
from converters import convert
import numpy as np
from gna.env import env

mat = np.matrix((np.arange(12).reshape((3,4))))
points = convert(mat, ROOT.Points)
print("Matrix\n", points.points.data())

tr = ROOT.Transpose()
tr.transpose.mat(points)
print(tr.transpose.T.data())
assert np.allclose(mat.T, tr.transpose.T.data()), "Python and C++ transpose doesn't coincide"