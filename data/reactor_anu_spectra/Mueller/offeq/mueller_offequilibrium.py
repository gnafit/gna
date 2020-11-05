#!/usr/bin/env python

import numpy as N
from matplotlib import pyplot as P
from collections import OrderedDict
from scipy.interpolate import interp1d

energy = N.concatenate([[1.8], N.linspace( 2., 4.5, 6 )])
"""First and last poinst are added ad-hoc"""
offeq = OrderedDict(
    U235  = N.array([ 0.060, 0.057, 0.044, 0.015, 0.007, 0.001,  0.0]),
    Pu239 = N.array([ 0.022, 0.021, 0.017, 0.005, 0.001, 0.0005, 0.0]),
    Pu241 = N.array([ 0.020, 0.019, 0.015, 0.005, 0.001, 0.0005, 0.0]),
)

newx = N.arange(1.8, 4.5, 0.01, dtype='d')

filename = 'mueller_offequilibrium_corr_{}.dat'

for name, data in offeq.items():
    f  = interp1d(energy, data, bounds_error=False, fill_value='extrapolate', kind='quadratic')
    newdata  = f(newx)
    newdata[newdata<0.0]=0.0

    fname = filename.format(name)
    N.savetxt( fname, N.transpose( [newx, newdata] ), fmt='%.3f %.12e' )
    print( 'Write', fname )

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'Enu' )
    ax.set_ylabel( 'Correction' )
    ax.set_title( 'Off-equilibrium correction for %s'%name )

    ax.plot( energy, data, 'o', label='data' )
    ax.plot( newx, newdata, '-', label='interpolation' )

    ax.legend(loc='upper right')

P.show()
