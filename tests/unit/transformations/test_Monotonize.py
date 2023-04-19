#!/usr/bin/env python

"""Check the Monotonize transformation"""

from matplotlib import pyplot as plt
import numpy as np
from gna import constructors as C
import pytest
from mpl_tools.helpers import savefig
import os
from gna.unittest import allure_attach_file

@pytest.mark.parametrize("direction", [+1, -1])
@pytest.mark.parametrize("gradient", [0, 0.5])
@pytest.mark.parametrize("start", [0, 0.5])
# @pytest.mark.parametrize("passx", [True, False])
def test_monotonize(direction, gradient, start, tmp_path): #,passx
    x = np.linspace(0.0, 10, 101)[1:]
    y = np.log(x)
    mask = x<1.0
    y[mask] = (x[mask]-1.0)**2

    if start==0:
        mask = x>1.0
    elif start>0.4:
        mask = x<1.0
    else:
        assert False

    if direction<0:
        y = -y

    # if not passx:
    #     x = np.arange(len(x))

    X = C.Points(x)
    Y = C.Points(y)

    # if passx:
    m = C.Monotonize(X, Y, start, gradient)
    # else:
    #     m = C.Monotonize(Y, start, gradient)

    ym = m.monotonize.yout.data()

    xmod  = x[mask]
    ymod  = ym[mask]
    ykept = ym[~mask]

    ymod_grad = (ymod[1:]-ymod[:-1])/(xmod[1:]-xmod[:-1])
    # print(direction, gradient, start, ymod_grad, np.fabs(ymod_grad)-gradient)

    assert (ykept==y[~mask]).all(), 'Modified the region, that should not be modified'
    assert np.allclose(np.fabs(ymod_grad), gradient, atol=1.e-14, rtol=0), 'Gradient of the modified region is not correct'
    if gradient!=0:
        diff = ym[1:]-ym[:-1]
        assert ((diff>0)==(diff[0]>0)).all(), 'The result is not monotonous'

    plt.figure()
    ax = plt.subplot(111, xlabel='x', ylabel='y', title=f'Grad {gradient}, start {start}')
    ax.grid()
    ax.plot(x, y, '+', label='input')
    ax.plot(x, ym, 'x', label='output')
    ax.legend()

    path = os.path.join(str(tmp_path), f'monotonize.png')
    savefig(path, dpi=300)
    allure_attach_file(path)
