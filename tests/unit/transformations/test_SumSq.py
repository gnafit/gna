#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *
import pytest

sumsq_mat = np.arange(12, dtype='d').reshape((3,4))

@pytest.mark.parametrize('arrays',
        [
         (3.0*sumsq_mat, 4.0*sumsq_mat,),
         (3.0*sumsq_mat, 4.0*sumsq_mat, 5.0*sumsq_mat),
         ]
        )
def test_sumsq_01(arrays):
    print('Test ', len(arrays), ':', sep='')
    for array in arrays:
        print(array)
    print()

    truth=sum(a*a for a in arrays)

    points = [C.Points(array) for array in arrays]
    sumsq = C.SumSq(outputs=[p.points.points for p in points])

    calc =  sumsq.single().data()
    print('Result', calc, end='\n\n')

    assert (calc==truth).all()

if __name__ == "__main__":
    glb = globals()
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('Run ', fcn)
        glb[fcn]()
