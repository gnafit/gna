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
par1 = env.defparameter( 'ad1.'+parname,  central=1.5, relsigma=0.1 )
par2 = env.defparameter( 'ad2.'+parname,  central=0.5, relsigma=0.1 )
(esmear1,esmear2), _ = detector_iav_from_file( 'output/detector_iavMatrix_P14A_LS.root', 'iav_matrix', ndiag=1,
                                               parname=parname,
                                               namespaces=[env.globalns(s) for s in ['ad1', 'ad2']] )

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
esmear1.smear.inputs.Ntrue( hist.hist )
esmear2.smear.inputs.Ntrue( hist.hist )

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

smeared1 = esmear1.smear.Nvis.data()
smeared2 = esmear2.smear.Nvis.data()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared1.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared2.sum() ) )

lines = plot_hist( edges, smeared1, linewidth=1.0, label='IAV 1' )
lines = plot_hist( edges, smeared2, linewidth=1.0, label='IAV 2' )

ax.legend( loc='upper right' )

P.show()
