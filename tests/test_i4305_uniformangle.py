#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as N
from load import ROOT as R
from matplotlib.ticker import MaxNLocator
R.GNAObject

p = R.UniformAngleParameter('double')('testangle')
p.set(0.0)

lim = 6
x1 = N.arange(   -lim*N.pi, +lim*N.pi, 0.1*N.pi, dtype='d' )
x2 = N.linspace( -lim*N.pi, +lim*N.pi, 121,      dtype='d'  )

def setter( v ):
    p.set( v )
    return p.value()

setter_v = N.vectorize( setter )
y1 = setter_v( x1 )
y2 = setter_v( x2 )
nerr = (y1>=N.pi).sum() + (y1<-N.pi).sum() + (y2>=N.pi).sum() + (y2<-N.pi).sum()

fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( r'Angle unconstrained, $n\pi$' )
ax.set_ylabel( r'Angle, $n\pi$' )
ax.set_title( 'Angle domain check (%s outliers)'%( nerr and 'FAIL: '+str(nerr) or 'OK: no' ) )

ax.plot( x1/N.pi, y1/N.pi, '^', markerfacecolor='none', label='arange' )
ax.plot( x2/N.pi, y2/N.pi, 'v', markerfacecolor='none', label='linspace' )

ax.xaxis.set_major_locator( MaxNLocator( nbins=30, steps=[1,2] ) )
ax.legend( loc='upper right' )
ax.set_ylim( top=1.4 )

plt.show()

if nerr:
    print( 'Error, there were %i outliers'%nerr )
