#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundle
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist
from gna.env import env
import constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from physlib import percent

#
# Initialize bundle
#
cfg = NestedDict(
        bundle = 'detector_iav_db_root_v01',
        parname = 'OffdiagScale',
        scale   = uncertain(1.0, 4, 'percent'),
        ndiag = 1,
        filename = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
        matrixname = 'iav_matrix'
        )
b, = execute_bundle( cfg=cfg )
esmear, = b.transformations_out.values()
par = b.common_namespace['OffdiagScale']

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
lines = plot_hist( edges, smeared2, linewidth=1.0, label='diag scale $s=2$' )

ax.legend( loc='upper right' )

P.show()
