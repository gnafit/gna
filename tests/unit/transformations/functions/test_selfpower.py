#!/usr/bin/env python3

"""Check the SelfPower transformation"""

from matplotlib import pyplot as plt
import numpy as np
from load import ROOT as R
from gna import constructors as C
from gna.env import env
import sys

def test_selfpower():
    #
    # Initialize parameters and data
    #
    ns = env.globalns('test_selfpower')
    varname = 'scale'
    par=ns.defparameter( varname, central=1.0, sigma=0.1 )

    arr = np.linspace( 0.0, 4.0, 81 )
    print( 'Data:', arr )

    #
    # Initialize transformations
    #
    points = C.Points( arr )
    with ns:
        selfpower = C.SelfPower( varname )

    selfpower.selfpower.points(points.points)
    selfpower.selfpower_inv.points(points.points)

    checks = ()

    #
    # Plot results
    # (Plot for positive power)
    #
    if not "pytest" in sys.modules:
        _ = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '$x$' )
        ax.set_ylabel( 'f(x)' )
        ax.set_title( 'SelfPower: $(x/a)^{x/a}$' )
    else:
        ax = None

    #
    # a=1
    #
    par.set(1)
    data1 = selfpower.selfpower.result.data().copy()

    if ax:
        ax.plot( arr, data, label='$a=1$' )

    checks1=(arr/par.value())**(arr/par.value())

    #
    # a=2
    #
    par.set(2)
    data2 = selfpower.selfpower.result.data().copy()

    if ax:
        ax.plot( arr, data, label='$a=2$' )
        ax.legend(loc='upper left')

    checks2=(arr/par.value())**(arr/par.value())

    #
    # Plot for negative power
    #
    if ax:
        _ = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( '$x$' )
        ax.set_ylabel( 'f(x)' )
        ax.set_title( 'SelfPower: $(x/a)^{-x/a}$' )

    #
    # a=1
    #
    par.set(1)
    data3 = selfpower.selfpower_inv.result.data().copy()
    checks3=(arr/par.value())**(-arr/par.value())

    if ax:
        ax.plot( arr, data, label='$a=1$' )

    #
    # a=2
    #
    par.set(2)
    data4 = selfpower.selfpower_inv.result.data().copy()

    if ax:
        ax.plot( arr, data, label='$a=2$' )
        ax.legend(loc='upper right')
        plt.show()

    checks4=(arr/par.value())**(-arr/par.value())

    for data, check in zip((checks1, checks2, checks3, checks4), (data1, data2, data3, data4)):
        assert np.allclose(check, data, atol=1.e-16, rtol=0)

if __name__ == "__main__":
    test_selfpower()
