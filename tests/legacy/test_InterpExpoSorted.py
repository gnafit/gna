#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.constructors import Points
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-U', '--underflow', default="", choices=['constant', 'extrapolate'] )
parser.add_argument( '-O', '--overflow', default="", choices=['constant', 'extrapolate'] )
parser.add_argument( '-o', '--output' )
opts = parser.parse_args()

segments   = N.arange(1.0, 10.1, 1.5, dtype='d')
# segments   = N.arange(1.0, 10.1, 4, dtype='d')
segments_t = Points(segments)

points = N.linspace(0.0, 12.0, 61)
points_t = Points(points)

fcn = N.exp( -(segments-segments[0])*0.5 )
fcn = N.exp(segments**(-0.5))
fcn_t = Points(fcn)

print( 'Edges', segments )
print( 'Points', points )
print( 'Fcn', fcn )

ie = R.InterpExpoSorted(opts.underflow, opts.overflow)
ie.interpolate(segments_t, fcn_t, points_t)
seg_idx = ie.segments.segments.data()
print( 'Segments', seg_idx )

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
# ax.set_yscale('log')

savefig(opts.output)

if opts.show:
    P.show()
