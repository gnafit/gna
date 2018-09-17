#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from constructors import Points
from gna.bindings import DataType

segments   = N.arange(0.0, 5.1, dtype='d')
segments_t = Points(segments)

N.random.seed(1)

print( 'Edges', segments )
for case in [
        ( 0.5, 3.6, 0.4),
        (-1.5, 3.6, 0.4),
        ( 3.5, 6.6, 0.4),
        (-1.5, 6.6, 0.4),
        (0.0, 5.1),
        (-1.e-17, 5.1),
        (-0.5, 6.5, 2.)
        ]:
    points   = N.arange(*case, dtype='d')
    print( '  points', points )
    N.random.shuffle(points)
    points_t = Points( points )

    sw = R.InSegment()
    sw.segments.edges(segments_t.points.points)
    sw.segments.points(points_t.points.points)

    res = sw.segments.insegment.data()
    print('  insegment', res)
    dgt = N.digitize( points, segments-1.e-16 )
    print('  digitize', dgt-1)

    comp = dgt==res
    print( (comp==0).all() and '\033[32mOK!\033[0m' or '\033[31mFAIL!\033[0m' )


    print()


