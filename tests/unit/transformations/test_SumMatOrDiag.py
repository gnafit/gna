#!/usr/bin/env python3

import gna.constructors as C
import numpy as np
import pytest

n = 4
mat = np.arange(1, n*n+1, dtype='d').reshape((n,n))
vec = np.arange(1, n+1, dtype='d')

@pytest.mark.parametrize('arrays',
        [(mat,),
         (mat, mat*2, mat*3,),
         (vec,),
         (vec, vec*2, vec*3,),
         (mat, vec,),
         (mat, vec, mat*3, mat*2, vec*3),
        ])
def test_sum_mat_or_diag_01(arrays):
    print('Test inputs', ':', sep='')
    for array in arrays:
        print(array)
    print()

    truth = np.zeros(n)
    for array in arrays:
        if array.shape == truth.shape:
            truth += array
        elif len(array.shape) == 1 and len(truth.shape) == 2:
            np.fill_diagonal(truth, truth.diagonal() + array)
        else: # len(array.shape) == 2 and len(truth.shape) == 1
            truth = np.diag(truth) + array

    print('Truth:\n', truth, end='\n\n')

    points = [C.Points(array) for array in arrays]
    sum = C.SumMatOrDiag([p.points.points for p in points])

    res =  sum.sum.sum.data()
    print('Result:\n', res, end='\n\n')

    assert (res==truth).all()
