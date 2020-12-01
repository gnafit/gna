#!/usr/bin/env python

"""Check Histogram class"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import stdvector, Histogram
from gna.bindings import DataType
from mpl_tools import bindings

def test_hist():
    #
    # Create the an array and bin edges
    #
    edges = N.arange(1.0, 7.1, 0.5)
    arr = N.arange(edges.size-1)

    print( 'Input edges and array (numpy)' )
    print( edges )
    print( arr )
    print()

    #
    # Create transformations
    #
    hist = Histogram(edges, arr)
    identity = R.Identity()
    identity.identity.source(hist.hist.hist)

    res = identity.identity.target.data()
    dt  = identity.identity.target.datatype()

    assert N.allclose(edges, dt.edges)
    assert N.allclose(arr, res)

    print( 'Eigen dump (C++)' )
    identity.dump()
    print()

    print( 'Result (C++ Data to numpy)' )
    print( res )
    print()

    print( 'Datatype:', str(dt) )
    print( 'Edges:', list(dt.edges) )

    if __name__ == "__main__":
        fig = plt.figure()
        ax = plt.subplot( 111 )
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel( 'x label' )
        ax.set_ylabel( 'entrie' )
        ax.set_title( 'Histogram' )
        identity.identity.target.plot_hist()

        plt.show()

if __name__ == "__main__":
    test_hist()
