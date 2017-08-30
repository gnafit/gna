#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from matplotlib import pyplot as P
from mpl_tools import bindings

h1 = R.TH1D( 'h1', 'gaus', 75, -5.0, 10.0 )
h2 = R.TH1D( 'h2', 'landau', 75, -5.0, 10.0 )
h3 = R.TH1D( 'h3', 'expo', 75, -5.0, 10.0 )

h1.FillRandom( 'gaus', 10000 )
h2.FillRandom( 'landau', 10000 )
h3.FillRandom( 'expo', 5000 )

#
# Plot with lines
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'Title' )

h1.plot( autolabel=True )
h2.plot( autolabel=True )
h3.plot( autolabel=True )

ax.legend( loc='upper left' )
ax.set_ylim( bottom=0.0 )

#
# Plot with errorbars
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'Title' )

h1.errorbar( autolabel=True )
h2.errorbar( autolabel=True )
h3.errorbar( autolabel=True )

ax.legend( loc='upper left' )

#
# Plot with bars
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'Title' )

h1.bar( autolabel=True, alpha=0.5 )
h2.bar( autolabel=True, alpha=0.5 )
h3.bar( autolabel=True, alpha=0.5 )

ax.legend( loc='upper left' )

#
# Plot with bars
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'Title' )

h1.bar( autolabel=True, divide=3, shift=0 )
h2.bar( autolabel=True, divide=3, shift=1 )
h3.bar( autolabel=True, divide=3, shift=2 )

ax.legend( loc='upper left' )

P.show()
