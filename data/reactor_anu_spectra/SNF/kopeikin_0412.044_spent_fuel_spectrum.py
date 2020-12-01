#!/usr/bin/env python

from scipy.signal import savgol_filter
from matplotlib import pyplot as P
import numpy as N

fname = './kopeikin_0412.044_spent_fuel_spectrum.dat'
oname = './kopeikin_0412.044_spent_fuel_spectrum_smooth.dat'
data = N.loadtxt(fname, unpack=True)
x, y = data

y_smooth = savgol_filter(y, 9, 3)

print( 'Read:', fname )
print( 'Write:', oname )
N.savetxt( oname, N.transpose([x, y]) )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'Enu, MeV' )
ax.set_ylabel( 'Ratio' )
ax.set_title( 'SNF to reactor ratio' )

ax.plot( x, y, 'o', markerfacecolor='none', label='Kopeikin et al.' )
ax.plot( x, y_smooth, label='smooth' )

ax.legend( loc='lower left' )

P.show()
