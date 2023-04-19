#!/usr/bin/env python

import load
import gna.constructors as C
import pytest
import numpy as np

@pytest.mark.parametrize('fcn', ['sin', 'cos', 'exp', 'sqrt'])
@pytest.mark.parametrize('passinput', [False, True])
def test_functions(fcn: str, passinput: bool):
    """Test various functions"""
    size=12
    functions = {
            'sin': C.Sin,
            'cos': C.Cos,
            'exp': C.Exp,
            'sqrt': C.Sqrt,
            }

    inarray=np.arange(size, dtype='d')
    array = C.Points(inarray)

    checkfcn = getattr(np, fcn)
    Cls = functions[fcn]
    if passinput:
        instance = Cls(array.single())
    else:
        instance = Cls()
        array >> instance.transformations[0].inputs[0]

    out = instance.transformations[0].outputs[0]
    ret=out.data()
    check = checkfcn(inarray)

    print(f'{fcn=} ({passinput=}): {inarray=}')
    print(f'{ret=}')
    print(f'{check=}')
    tol = np.finfo('d').resolution*10000
    assert np.allclose(ret, check, atol=tol, rtol=0)

