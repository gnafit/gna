#!/usr/bin/env python

from load import ROOT as R
import numpy as N
from gna.constructors import Points
from gna.bindings import DataType

segments   = N.arange(0.0, 5.1, dtype='d')
segments_t = Points(segments)

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
    points_t = Points( points )

    sw = R.SegmentWise()
    sw.segments.edges(segments_t.points.points)
    sw.segments.points(points_t.points.points)

    res = sw.segments.segments.data()
    print('  segments', res)
    for i, (i1, i2) in enumerate(zip(res[:-1], res[1:])):
        i1, i2 = int(i1), int(i2)
        sub = points[i1:i2]
        xmin, xmax = segments[i], segments[i+1]
        check = (xmin<=sub)*(sub<xmax)
        msg = '\033[31mFAIL!\033[0m' if not check.all() else ''
        print('    %i %g->%g:'%(i, xmin, xmax), sub, msg )
    print()


