#!/usr/bin/env python

"""Check the ArrayGradient2p transformation"""

from matplotlib import pyplot as plt
import numpy as np
from gna import constructors as C
import pytest
from mpl_tools.helpers import savefig
import os
from gna.unittest import allure_attach_file

def test_monotonize(tmp_path):
    x = np.geomspace(1, 10, 20)
    y = np.log(x)

    X = C.Points(x)
    Y = C.Points(y)
    g = C.ArrayGradient2p(X, Y)

    checkg = (y[1:]-y[:-1])/(x[1:]-x[:-1])
    checkx = (x[1:]+x[:-1])*0.5;
    retx = g.gradient.xout.data()
    retg = g.gradient.gradient.data()

    assert np.allclose(retx, checkx, atol=1e-15, rtol=0)
    assert np.allclose(retg, checkg, atol=1e-15, rtol=0)

    plt.figure()
    ax = plt.subplot(111, xlabel='', ylabel='', title=f'gradient')
    ax.grid()
    ax.plot(x, y, 'o', markerfacecolor='none', label='function')

    xfine = np.linspace(x[0], x[-1], 200)
    ax.plot(xfine, 1/xfine, '-', alpha=0.5, label='analytic gradient')
    ax.plot(retx, retg, 'o', markerfacecolor='none', label='transformation')
    ax.legend()

    path = os.path.join(str(tmp_path), f'gradient.png')
    savefig(path, dpi=300)
    allure_attach_file(path)

    # plt.show()
