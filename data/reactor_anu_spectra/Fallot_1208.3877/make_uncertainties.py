#!/usr/bin/env python

from numpy import *

# What I told to Double Chooz as regards U8 spectrum which should be the most
# uncertain of the 4 spectra is to consider a 10% systematic error up to 5 MeV,
# 15% between 5 and 6 MeV and 20% above 6 MeV.
# Regarding 235U a 10% global envelop up to 8 MeV should be conservative, as it
# is the best known, and for 239,241Pu, a 10% error up to 5.5MeV, 15% between
# 5.5 and 7MeV and 20% above 7MeV.

percent = 0.01
uncertainties = dict(
    U235 = { 8  : 10.*percent,
             20 : 20.*percent },
    U238 = { 5  : 10.*percent,
             6  : 15.*percent,
             20 : 20.*percent },
    Pu239 = { 5.5 : 10.*percent,
              7  : 15.*percent,
              20 : 20.*percent },
)
uncertainties[ 'Pu241' ] = uncertainties[ 'Pu239' ]

e = arange( 1.8, 10.00000001, 0.2 )
e = 0.5*( e[1:]+e[:-1] )

header = '''Uncertainties for %s Fallot's spectrum from arXiv:1208.3877
Bin center [MeV], total relative uncertainty'''

for iso, unc_limits in uncertainties.iteritems():
    unc = zeros( len(e), dtype='d' )
    for lim in reversed( sorted( unc_limits.keys() ) ):
        unc[ e<lim ] = unc_limits[lim]

    a = array( [e, unc] ).T
    fname = 'fallot_%s_uncertainty.dat'%iso
    savetxt( fname, a, fmt='%.2f', header=header%iso )
    print( 'Write', fname )
