#!/usr/bin/env python
import load
import ROOT
from gna.converters import convert
import numpy as np
from gna.env import env

def test_mat_mult():
    mat1 = np.arange(9).reshape((3,3))
    points1 = convert(mat1, ROOT.Points)
    print("First matrix\n", points1.points.data())

    mat2 = (np.arange(12).reshape((3,4)))
    points2 = convert(mat2, ROOT.Points)
    print("Second matrix\n", points2.points.data())

    prod = ROOT.MatrixProduct()
    prod.multiply(points1)
    prod.multiply(points2)
    prod_of_two = np.matmul(mat1, mat2)
    print("Product of two from python\n", prod_of_two )
    print("Product of two from C++\n", prod.product.data())
    assert np.allclose(prod_of_two, prod.product.data()), "The products of two matrices don't coincide"

    print("Add third matrix to product")

    prod2 = ROOT.MatrixProduct()
    mat3 = np.arange(10, 30).reshape((4,5))
    points3 = convert(mat3, ROOT.Points)
    prod2.multiply(points1)
    prod2.multiply(points2)
    prod2.multiply(points3)
    prod_of_three = np.matmul(prod_of_two, mat3)
    print("Third matrix\n", points3.points.data())
    print("Product of three from python\n", prod_of_three)
    print("Product of three from C++\n", prod2.product.data())
    assert np.allclose(prod_of_three, prod2.product.data()), "The products of three matrices don't coincide"

if __name__ == "__main__":
    test_mat_mult()
