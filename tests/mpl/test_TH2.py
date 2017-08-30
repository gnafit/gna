#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from matplotlib import pyplot as P
from mpl_tools import bindings

xyg=R.TF2("xyg","xygaus", -10, 10, -10, 10)
xyg.SetParameters(1, 0, 1, 0, 4)
R.gDirectory.Add( xyg )

h2 = R.TH2D( 'h2', 'gaus', 50, -5.0, 5.0, 100, -10.0, 10.0 )
h2.FillRandom( 'xyg', 100000 )

#
# pcolorfast (as is)
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'pcolorfast' )

h2.pcolorfast( colorbar=True )

#
# pcolorfast (mask zero)
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'pcolorfast' )

h2.pcolorfast( colorbar=True, mask=0.0 )

#
# pcolormesh (mask zero)
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'pcolormesh' )

h2.pcolormesh( colorbar=True, mask=0.0 )

#
# imshow (mask zero)
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'imshow' )

h2.imshow( colorbar=True, mask=0.0 )

P.show()
