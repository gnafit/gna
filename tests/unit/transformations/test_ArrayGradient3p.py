#!/usr/bin/env python

"""Check the ArrayGradient3p transformation"""

from matplotlib import pyplot as plt
import numpy as np
from gna import constructors as C
import pytest
from mpl_tools.helpers import savefig
import os
from gna.unittest import allure_attach_file

@pytest.mark.parametrize('step', ['constant', 'variable'])
def test_monotonize(step, tmp_path):
    if step=='constant':
        x = np.linspace(1, 10, 20)
    elif step=='variable':
        x = np.geomspace(1, 10, 20)
    else:
        assert False
    y = np.log(x)

    X = C.Points(x)
    Y = C.Points(y)
    g = C.ArrayGradient3p(X, Y)

    check = np.gradient(y, x)
    ret = g.gradient.gradient.data()

    assert np.allclose(ret, check, atol=1e-15, rtol=0)

    # plt.figure()
    # ax = plt.subplot(111, xlabel='', ylabel='', title=f'Function')
    # ax.grid()
    # ax.plot(x, y, 'o', markerfacecolor='none')

    plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title=f'gradient')

    sg = (y[1:]-y[:-1])/(x[1:]-x[:-1])
    ax.plot(0.5*(x[1:]+x[:-1]), sg, 'o', markerfacecolor='none', label='simple gradient')

    xfine = np.linspace(x[0], x[-1], 200)
    ax.plot(xfine, 1/xfine, '-', alpha=0.5, label='function')

    ax.plot(x, ret, '-', label='transformation')
    ax.plot(x, check, '--', label='np')

    ax.grid()
    ax.legend()

    path = os.path.join(str(tmp_path), f'gradient.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

