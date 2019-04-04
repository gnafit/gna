#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
import gna.constructors as C

def test_histedges():
    edges = N.linspace(0.0, 10.0, 11)
    data  = N.arange(10.0)

    hist = C.Histogram( edges, data )
    h2e = R.HistEdges()
    h2e.histedges.hist( hist.hist.hist )

    e = h2e.histedges.edges.data()

    print( 'Input:' )
    print( edges )

    print( 'Output:' )
    print( e )

    print( (edges==e).all() and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )
    assert (edges==e).all()

if __name__ == "__main__":
    test_histedges()
