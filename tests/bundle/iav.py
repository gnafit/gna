#!/usr/bin/env python

from load import ROOT as R
R.GNAObject
from gna.bundle import execute_bundles
from matplotlib import pyplot as P
from matplotlib.colors import LogNorm
from mpl_tools.helpers import add_colorbar, plot_hist, savefig
from gna.bindings import common
from gna.env import env
import gna.constructors as C
import numpy as N
from gna.configurator import NestedDict, uncertain
from physlib import percent

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument( '-o', '--output', help='output file' )
parser.add_argument( '-x', '--xlim', nargs=2, type=float, help='xlim' )
args = parser.parse_args()

#
# Initialize bundle
#
cfg = NestedDict(
    # Bundle name
    bundle = dict(name='detector_iav_db_root', version='v03',
        nidx=[('d', 'detector', ['D1'])],
        ),
    # Parameter name
    parname = 'OffdiagScale',
    # Parameter uncertainty and its type (absolute or relative)
    scale   = uncertain(1.0, 4, 'percent'),
    # Number of diagonals to treat as diagonal. All other elements are considered as off-diagonal.
    ndiag = 1,
    # File name to read
    filename = 'data/data_dayabay/tmp/detector_iavMatrix_P14A_LS.root',
    # Matrix name
    matrixname = 'iav_matrix'
    )
b, = execute_bundles( cfg=cfg )
smear = b.context.outputs.iav.D1
b.namespace.printparameters()
par = b.namespace['OffdiagScale.D1']

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

phist = singularities( [4.025, 7.025, 9.025, 11.025 ], edges )
hist = C.Histogram( edges, phist )
hist.hist >> b.context.inputs.iav.D1['00']

#
# Plot
#
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( r'$E_\nu$, MeV' )
ax.set_ylabel( 'entries' )
ax.set_title( 'IAV effect' )

smeared = smear.data().copy()
par.set( 2.0 )
smeared2 = smear.data().copy()
print( 'Sum check for {} (diff): {}'.format( 1.0, phist.sum()-smeared.sum() ) )
print( 'Sum check for {} (diff): {}'.format( 2.0, phist.sum()-smeared2.sum() ) )

lines = plot_hist( edges, smeared, label='nominal' )
lines = plot_hist( edges, smeared2, linewidth=1.0, label='diag scale $s=2$' )

ax.legend( loc='upper right' )

if args.xlim:
    ax.set_xlim( *args.xlim )

savefig( args.output )

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( 'IAV matrix' )

from matplotlib.colors import LogNorm
b.context.outputs.iavmatrix.D1.plot_matshow(colorbar=True, norm=LogNorm())

savefig( args.output, suffix='_mat' )

#
# Dump graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot( b.transformations_out.values()[0] )
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )

if args.show:
    P.show()
