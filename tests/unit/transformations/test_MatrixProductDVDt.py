#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import matplotlib as mpl
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
import pytest
from gna.unittest import allure_attach_file, savegraph

from load import ROOT as R
from gna import constructors as C

@pytest.mark.parametrize('n1', range(1,8,3))
@pytest.mark.parametrize('n2', range(1,8,3))
def test_MatrixProductDVDt(n1, n2, tmp_path):
    '''Matrix product of D V Dt'''

    size1, size2 = n1*n2, n2*n2
    left_n   = np.arange(size1, dtype='d').reshape(n1,n2) + 1.0
    square_n = 40.0+size2 - np.arange(size2, dtype='d').reshape(n2, n2)

    left = C.Points(left_n)
    square = C.Points(square_n)

    dvdt = R.MatrixProductDVDt(left, square)

    result = dvdt.product.product()
    check  = np.matmul(np.matmul(left_n, square_n), left_n.transpose())
    diff   = np.fabs(result - check)
    # print('Max diff ({}, {})'.format(n1, n2), diff.max())
    assert (result==check).all()

    path = os.path.join(str(tmp_path), 'graph_{:d}_{:d}.png'.format(n1, n2))
    savegraph(dvdt.product, path, verbose=False)
    allure_attach_file(path)

