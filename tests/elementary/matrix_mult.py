#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
from converters import convert
import numpy as np
from gna.env import env

mat1 = np.matrix(np.arange(9).reshape((3,3)))
points1 = convert(mat1, ROOT.Points)
print("First matrix\n", points1.points.data())

mat2 = np.matrix((np.arange(12).reshape((3,4))))
points2 = convert(mat2, ROOT.Points)
print("Second matrix\n", points2.points.data())

prod = ROOT.MatrixProduct()
prod.multiply(points1)
prod.multiply(points2)
print("From python\n", np.dot(mat1, mat2))
print("From C++\n", prod.product.data())
assert(np.allclose(np.dot(mat1, mat2), prod.product.data()), "The products
don't coincide")
print("Test passed")
