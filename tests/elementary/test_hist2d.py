#!/usr/bin/env python

"""Check Histogram class"""

from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
from gna.constructors import stdvector, Histogram2d
from gna.bindings import DataType
from mpl_tools import bindings

def test_hist():
    #
    # Create the an array and bin edges
    #
    xedges = N.arange(1.0, 7.1, 0.5)
    yedges = N.arange(3.0, 9.1, 1.5)
    arr = N.arange((xedges.size-1)*(yedges.size-1)).reshape(xedges.size-1, yedges.size-1)

    print( 'Input edges and array (numpy)' )
    print( 'X', xedges )
    print( 'Y', yedges )
    print( arr )
    print()

    #
    # Create transformations
    #
    hist = Histogram2d(xedges, yedges, arr)

    res = hist.hist.hist.data()
    dt  = hist.hist.hist.datatype()

    assert N.allclose(xedges, dt.edges)
    assert N.allclose(xedges, dt.edgesNd[0])
    assert N.allclose(yedges, dt.edgesNd[1])
    assert N.allclose(arr, res)

    print( 'Result (C++ Data to numpy)' )
    print( res )
    print()

    print( 'Datatype:', str(dt) )
    print( 'Edges:', list(dt.edgesNd[0]), list(dt.edgesNd[1]) )

if __name__ == "__main__":
    test_hist()
