#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import gna.constructors as C
import numpy as np
from gna.unittest import *

@clones(globals(), float=False, gpu=True)
def test_rebin():
    edges   = np.array( [ 0.0, 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.0 ], dtype='d' )
    edges_m = np.array( [      0.1, 1.2,      3.4, 4.5,           7.8      ], dtype='d' )
    expect  = np.array( [1.0, 2.0, 1.0, 3.0], dtype='d' )

    ntrue = C.Histogram(edges, np.ones( edges.size-1 ) )
    rebin = C.Rebin(edges_m, 3)
    ntrue >> rebin

    newdata = rebin.rebin.histout.data()

    from gna.converters import convert
    mat = convert(rebin.getDenseMatrix(), 'matrix')
    prj = mat.sum(axis=0)

    assert ((prj==1.0) + (prj==0.0)).all()
    assert (newdata==expect).all()
    assert (list(rebin.rebin.histout.datatype().edges)==edges_m).all()

if __name__ == "__main__":
    run_unittests(globals())
