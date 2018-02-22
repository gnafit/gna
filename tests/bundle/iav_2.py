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
from gna.configurator import NestedDict
from physlib import percent

#
# Initialize bundle
#
cfg = NestedDict(
        bundle = 'detector_iav_db_root_v01',
        parname = 'OffdiagScale',
        uncertainty = 10*percent,
        uncertainty_type = 'relative',
        ndiag = 1,
        filename = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
        matrixname = 'iav_matrix'
        )
b = execute_bundle( cfg=cfg, namespaces=['ad1', 'ad2', 'ad3'] )
(esmear1, esmear2, esmear3) = b.output_transformations

par1, par2, par3 = (b.common_namespace(s)['OffdiagScale'] for s in ('ad1', 'ad2', 'ad3'))
par1.set( 1.5 )
par2.set( 1.0 )
par3.set( 0.5 )

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
esmear3.smear.inputs.Ntrue( hist.hist )

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
smeared3 = esmear3.smear.Nvis.data()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared1.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared2.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared3.sum() ) )

lines = plot_hist( edges, smeared1, linewidth=1.0, label='IAV 1' )
lines = plot_hist( edges, smeared2, linewidth=1.0, label='IAV 2' )
lines = plot_hist( edges, smeared3, linewidth=1.0, label='IAV 3' )

ax.legend( loc='upper right' )

P.show()