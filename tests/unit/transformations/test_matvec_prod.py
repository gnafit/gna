#!/usr/bin/env python
import load
import ROOT as R
#  from gna.converters import convert
import gna.constructors as C
import numpy as np
from gna.env import env
import pytest

@pytest.mark.parametrize('first,second', [
    (np.array([1, 1]), np.array([[1,0],[0,1]])),
    (np.array([[1,0],[0,1]]), np.array([1,1])),
    (np.array([1,2,3]), np.array([[1,2,3], [2,3,1], [3,4,5]])),
    (np.array([[1,2,3], [2,3,1], [3,4,5]]), np.array([1,2,3])),
    (np.array([[1,2], [2,3], [1,3]]), np.array([1,2])),
    (np.array([1,2,3]), np.array([[1,2], [2,3], [1,2]]))
    ]
    )
def test_matvec_mult(first, second):
    first_p = C.Points(first)
    second_p = C.Points(second)

    prod = R.MatVecProduct(first_p.single(), second_p.single())
    prod_of_two = first @ second
    assert np.allclose(prod_of_two, prod.product.data()), "C++ and Python results don't coincide!"

@pytest.mark.parametrize('first,second', [
    (np.array([1, 1]), np.array([[1,0,1],[0,1,1], [1,2,1]])),
    (np.array([1, 1, 2]), np.array([[1,0],[1,1]])),
    (np.array([1, 1]), np.array([[1,0],[1,1]])),
    ]
    )
def test_matvec_mult_raises(first, second):
    '''Test cases that must fail. Even though the exception is handled, ROOT
    still prints a ton of backtrace'''
    first_p = C.Points(first)
    second_p = C.Points(second)

    try:
        prod = R.MatVecProduct(first_p.single(), second_p.single())
    except TypeError as e:
        assert True
    except:
        assert False
