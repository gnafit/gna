#!/usr/bin/env python

import ROOT as R
import numpy as N
from matplotlib import pyplot as P
from mpl_tools import bindings

x = N.arange( 0.0, 100.0, 5 )
y = (x-50.0)**2/100
g = R.TGraph( x.size, x, y )

y+=3
ye = y**0.5
xe = (x[2]-x[1])*0.5*N.ones_like(x)
ge = R.TGraphErrors( x.size, x, y, xe, ye )

y+=20
xel = (x[2]-x[1])*0.8*N.linspace( 1.0, 0.0, x.size)
xeh = (x[2]-x[1])*0.8*N.linspace( 0.0, 1.0, x.size)
yel = y**0.5
yeh = y**0.25
gea = R.TGraphAsymmErrors( x.size, x, y, xel, xeh, yel, yeh )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x axis' )
ax.set_ylabel( 'y axis' )
ax.set_title( 'graphs' )

g.plot( 'o-', label='TGraph.plot("o-")' )
ge.errorbar( fmt='o', markerfacecolor='none', label='TGraphErrors.errorbar(fmt="o")' )
ge.plot( '-', markerfacecolor='none', label='TGraphErrors.plot("-")' )
gea.errorbar( fmt='o-', markerfacecolor='none', label='TGraphAsymmErrors.errorbar(fmt="o-")' )

ax.legend( loc='upper center' )

P.show()

