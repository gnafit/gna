#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.constructors import Points
from matplotlib import pyplot as P
from mpl_tools.helpers import savefig
from gna.bindings import common

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-o', '--output' )
opts = parser.parse_args()

x = N.array([0.0, 2.0, 4.0, 6.0], dtype='d')
y = N.array([10.0, 8.0, 6.0, 4.0], dtype='d')
xp = Points(x)
yp = Points(y)

point = N.array([3.0], dtype='d')
pointp = Points(point)

scaler = R.FixedPointScale()
scaler.scale(xp, yp, pointp)
seg_idx = scaler.insegment.insegment.data()
print( 'Segments', seg_idx )

# res = ie.interp.interp.data()
# print( 'Result', res )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_title( 'Scaling' )

yp.points.points.plot_vs( xp.points.points, 'o-', markerfacecolor='none', label='initial function')
scaler.interp.interp.plot_vs( xp.points.points, 'o-', markerfacecolor='none', label='scaled function')
ax.axhline(1.0)

ax.legend(loc='upper right')

savefig(opts.output)


if opts.show:
    P.show()
