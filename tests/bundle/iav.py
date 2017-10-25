#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle.detector_iav import detector_iav_from_file
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist
from gna.env import env
import constructors as C
import numpy as N

#
# Initialize bundle
#
parname = 'DiagScale'
par = env.defparameter( parname,  central=1.0, relsigma=0.1 )
esmear, transf = detector_iav_from_file( 'output/iavMatrix_P14A_LS.root', 'iav_matrix', ndiag=4, parname=parname )

#
# Test bundle
#
def singularities( values, edges ):
    indices = N.digitize( values, edges )-1
    phist = N.zeros( edges.size-1 )
    phist[indices] = 1.0
    return phist

binwidth=0.05
edges = N.arange( 0.0, 12.0001, binwidth )

phist = singularities( [ 1.225, 4.025, 7.025 ], edges )
hist = C.Histogram( edges, phist )
esmear.smear.inputs.Ntrue( hist.hist )

#
# Plot
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'IAV effect' )

smeared = esmear.smear.Nvis.data().copy()
par.set( 2.0 )
smeared2 = esmear.smear.Nvis.data().copy()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared2.sum() ) )

lines = plot_hist( edges, smeared, label='nominal' )
lines = plot_hist( edges, smeared2, linewidth=1.0, label='s=2' )

ax.legend( loc='upper right' )

P.show()
