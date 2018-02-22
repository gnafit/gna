#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from constructors import Points
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-o', '--output' )
opts = parser.parse_args()

segments   = N.arange(1.0, 10.1, 1.5, dtype='d')
segments_t = Points(segments)

points = N.linspace(0.0, 12.0, 61)
points_t = Points(points)

fcn = N.exp( -(segments-segments[0])*0.5 )
fcn_t = Points(fcn)

print( 'Edges', segments )
print( 'Points', points )
print( 'Fcn', fcn )

ie = R.InterpExpo()
ie.segments.edges(segments_t)
ie.segments.points(points_t)
seg_idx = ie.segments.segments.data()
print( 'Segments', seg_idx )

ie.interp.points(points_t)
ie.interp.edges(segments_t)
ie.interp.fcn(fcn_t)
ie.interp.segments(ie.segments)

res = ie.interp.interp.data()
print( 'Result', res )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_title( 'Expo' )

ax.plot( segments, fcn, 'o', markerfacecolor='none', label='coarse function' )
ax.plot( points, res, '.', label='interpolation' )
ax.legend(loc='upper right')

savefig(opts.output)

if opts.show:
    P.show()
