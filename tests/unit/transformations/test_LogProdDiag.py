#!/usr/bin/env python

"""Check the LogProdDiag transformation"""

import numpy as np
from gna import constructors as C
import pytest

@pytest.mark.parametrize('type', ['full', 'tril', 'diag'])
@pytest.mark.parametrize('scale', [1.0, 2.0, None])
@pytest.mark.parametrize('nmats', [1, 2])
def test_LogProdDiag(type: str, scale: float, nmats: int):
    size = 16

    mats = [np.arange(size, dtype='d').reshape(4, 4)+1.0+i*i for i in range(nmats)]
    diags = [np.diag(mat) for mat in mats]

    if type=='tril':
        mats = [np.tril(mat) for mat in mats]
    elif type=='diag':
        mats = diags
    else:
        assert type=='full'

    print('Matrices (numpy)')
    print(mats)
    print()

    #
    # Create the transformations
    #
    if scale is not None:
        trn = C.LogProdDiag(scale)
    else:
        scale = 2.0
        trn = C.LogProdDiag()

    points = [C.Points(mat) for mat in mats]
    for p in points: trn.add(p)

    out=trn.logproddiag.logproddiag
    ret = out.data()[0]

    check1 = sum(scale*np.log(diag.prod()) for diag in diags)
    check2 = sum(scale*np.log(diag).sum() for diag in diags)
    check3 = None

    if type=='tril':
        check3 = sum(scale*np.log(np.linalg.det(mat)) for mat in mats)

    print(f'{scale=}', end=', ')
    print(f'{ret=}', end=', ')
    print(f'{check1=}', end=', ')
    print(f'{check2=}', end=', ')
    print(f'{check3=}')

    for c in (check1, check2, check3):
        if c is None: continue
        assert np.allclose(ret, c, rtol=0, atol=1.e-16)
