#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *
import pytest

sum_broadcast_mat_m11 = np.ones((1,), dtype='d')
sum_broadcast_mat_m34 = np.ones((3,4), dtype='d')
sum_broadcast_mat_m33 = np.ones((3,3), dtype='d')

sum_broadcast_rup   = np.arange(12.0, dtype='d').reshape(3,4)
sum_broadcast_rdown = np.arange(12.0, dtype='d')[::-1].reshape(3,4)

@pytest.mark.parametrize('arrays',
        [(sum_broadcast_mat_m11,),
         (sum_broadcast_mat_m11*2, sum_broadcast_mat_m11*3, sum_broadcast_mat_m11*4,),
         (sum_broadcast_mat_m11*2, sum_broadcast_mat_m11*0,),
         (sum_broadcast_mat_m34,),
         (sum_broadcast_mat_m34, sum_broadcast_mat_m34*0),
         (2.0*sum_broadcast_mat_m11, 3.0*sum_broadcast_mat_m34,),
         (3.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11,),
         (2.0*sum_broadcast_mat_m11, 3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34,),
         (3.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11, 4.0*sum_broadcast_mat_m34,),
         (3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m11,),
         (3.0*sum_broadcast_mat_m34, 4.0*sum_broadcast_mat_m34, 2.0*sum_broadcast_mat_m34,),
         (sum_broadcast_rup, sum_broadcast_rdown,)
         ])
def test_sum_broadcast(arrays):
    print('Test ', len(arrays), ':', sep='')
    for array in arrays:
        print(array)
    print()

    truth=0.0
    for a in arrays:
        if a.size==1:
            truth+=a[0]
        else:
            truth+=a

    print('Truth:\n', truth, end='\n\n')

    points = [C.Points(array) for array in arrays]
    sum = C.SumBroadcast(outputs=[p.points.points for p in points])

    calc =  sum.single().data()
    print('Result', calc, end='\n\n')

    assert (calc==truth).all()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('Run ', fcn)
        glb[fcn]()
