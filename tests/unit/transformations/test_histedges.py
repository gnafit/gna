#!/usr/bin/env python

from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
import gna.constructors as C
import pytest

@pytest.mark.parametrize('edges', [N.linspace(0.0, 10.0, 11), N.geomspace(0.1, 1000.0, 5)])
def test_histedges_v01(edges):
    centers = 0.5*(edges[1:]+edges[:-1])
    widths  =      edges[1:]-edges[:-1]
    data  = N.arange(edges.size-1)

    hist = C.Histogram(edges, data)
    h2e = R.HistEdges()
    h2e.histedges.hist(hist.hist.hist)

    out_edges   = h2e.histedges.edges.data()
    out_centers = h2e.histedges.centers.data()
    out_widths = h2e.histedges.widths.data()

    print( 'Input:' )
    print( edges )

    print( 'Output:' )
    print( 'Edges', out_edges )
    print( 'Centers', out_centers )
    print( 'Widths', out_widths )

    assert (edges==out_edges).all()
    assert (centers==out_centers).all()
    assert (widths==out_widths).all()

if __name__ == "__main__":
    test_histedges()
