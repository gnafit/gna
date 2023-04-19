#!/usr/bin/env python

from load import ROOT as R
import numpy as N
from gna.constructors import Points
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig

interpolators = dict(
        linear = R.InterpLinear,
        log = R.InterpLog,
        logx = R.InterpLogx,
        expo = R.InterpExpo,
        const = R.InterpConst
        )

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-m', '--mode', default='expo', choices=interpolators.keys(), help='Interpolation mode' )
parser.add_argument( '-r', '--repeat', nargs=3, type=float, default=[], action='append', metavar=('index', 'scale_left', 'scale_right'), help='repeat the particular edge and scale the value')
# parser.add_argument( '-U', '--underflow', default="", choices=['constant', 'extrapolate'] )
# parser.add_argument( '-O', '--overflow', default="", choices=['constant', 'extrapolate'] )
parser.add_argument( '-o', '--output' )
opts = parser.parse_args()

segments   = N.arange(1.0, 10.1, 1.5, dtype='d')
# fcn = N.exp( -(segments-segments[0])*0.5 )
fcn = N.exp(segments**(-0.5))

for idx, sleft, sright in opts.repeat:
    idx = int(idx)
    segments = N.insert(segments, idx, segments[idx])
    fcn      = N.insert(fcn, idx, sleft*fcn[idx])
    fcn[idx+1]*=sright

points   = N.stack([N.linspace(0.0+i, 12.+i, 61, dtype='d') for i in [0, -0.1, 0.1, 0.3, 0.5]]).T
if opts.mode=='logx':
    points=points[1:]

print( 'Edges', segments )
print( 'Points', points[:20] )
print( 'Fcn', fcn[:20] )

segments_t = Points(segments)
points_t = Points(points)
fcn_t = Points(fcn)
ie=interpolators[opts.mode]()
ie.interpolate(segments_t, fcn_t, points_t)
seg_idx = ie.insegment.insegment.data()
print( 'Segments', seg_idx[:20] )

ie.print()

res = ie.interp.interp.data()
nans = N.nonzero(N.isnan(res))
fnans = N.flatnonzero(N.isnan(res))
print( 'Result', res )
if fnans.size:
    print( '\033[31mNans', fnans, '\033[0m' )
    print( '\033[31mNan locations', points[nans], '\033[0m' )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_title( opts.mode.capitalize() )

ax.plot( segments, fcn, 'o-', markerfacecolor='none', label='coarse function', linewidth=0.5, alpha=0.5 )

markers='os*^vh'
for i, (p, r) in enumerate(zip(points.T, res.T)):
    ax.plot( p, r, '.', label='interpolation, col %i'%i, marker=markers[i], markersize=1.5 )
    break

if fnans.size:
    P.vlines(points[nans], 0, 10, linestyles='dashed', colors='red', label='nans')

ax.legend(loc='upper right')
# ax.set_yscale('log')

savefig(opts.output)

if opts.show:
    P.show()
