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
prod_of_two = np.dot(mat1, mat2)
print("Product of two from python\n", prod_of_two )
print("Product of two from C++\n", prod.product.data())
assert np.allclose(prod_of_two, prod.product.data()), "The products of two matrices don't coincide"

print("Add third matrix to product")

prod2 = ROOT.MatrixProduct()
mat3 = np.matrix((np.arange(10, 30).reshape((4,5))))
points3 = convert(mat3, ROOT.Points)
prod2.multiply(points1)
prod2.multiply(points2)
prod2.multiply(points3)
prod_of_three = np.dot(prod_of_two, mat3)
print("Third matrix\n", points3.points.data())
print("Product of three from python\n", prod_of_three)
print("Product of three from C++\n", prod2.product.data())
assert np.allclose(prod_of_three, prod2.product.data()), "The products of three matrices don't coincide"