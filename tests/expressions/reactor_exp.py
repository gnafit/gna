#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output' )
parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
parser.add_argument('-m', '--mode', default='small', choices=['complete', 'small', 'minimal'], help='Set the indices coverage')
args = parser.parse_args()

#
# Import libraries
#
from gna.expression import *
from gna.configurator import uncertaindict, uncertain
from gna.bundle import execute_bundle
from load import ROOT as R
from gna.env import env
from matplotlib import pyplot as P
import numpy as N
from mpl_tools import bindings
from gna.labelfmt import formatter as L
R.GNAObject

#
# Define the indices (empty for current example)
#
if args.mode=='complete':
    indices = [
        ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
        ('r', 'reactor',     ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
        ]
elif args.mode=='minimal':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1']),
        ('d', 'detector',    ['AD11']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
        ]
elif args.mode=='small':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1', 'LA1']),
        ('d', 'detector',    ['AD11', 'AD12']),
        ('c', 'component',   ['comp0', 'comp12'])
        ]
else:
    raise Exception('Unsupported mode '+args.mode)

lib = dict(
        cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob'),
        cspec_diff_reac         = dict(expr='sum:i'),
        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
        cspec_diff_det          = dict(expr='sum:r'),
        spec_diff_det           = dict(expr='sum:c'),
        cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),
        # norm_bf                 = dict(expr='eff*efflivetime*effunc_uncorr*global_norm'),
        norm_bf                 = dict(expr='eff*effunc_uncorr*global_norm'),
        )

expr =[
        'baseline[d,r]',
        'enu| ee(evis()), ctheta()',
        'jacobian| enu(), ee(), ctheta()',
        'ibd_xsec(enu(), ctheta())',
        'oscprob[c,d,r]( enu() )',
        'anuspec[i](enu())',
        'eres_matrix| evis_edges()',
        'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
        'power_livetime_factor_daily = efflivetime_daily[d]()*thermal_power[r]()*fission_fractions[i,r]()',
        'power_livetime_factor=accumulate("power_livetime_factor", power_livetime_factor_daily)',
        'livetime=accumulate("livetime", livetime_daily[d]())',
        '''result = global_norm *  eff * effunc_uncorr[d] *
                      eres[d]|
                      iav[d] |
                      sum[c]| pmns[c]*
                        sum[r]|
                          baselineweight[r,d]*
                          sum[i]|
                            power_livetime_factor*
                            kinint2|
                              anuspec() * oscprob() * ibd_xsec() * jacobian()''',
        ]

# Initialize the expression and indices
a = Expression(expr, indices)

# Dump the information
print(a.expressions_raw)
print(a.expressions)

# Parse the expression
a.parse()
# The next step is needed to name all the intermediate variables.
a.guessname(lib, save=True)
# Dump the tree.
a.tree.dump(True)

#
# At this point what you have is a dependency tree with variables, transformations (all indexed),
# but without actual implementation. We add the implementation on a next step.
#

print()
# Here is the configuration
cfg = NestedDict(
        kinint2 = NestedDict(
            bundle   = 'integral_2d1d_v01',
            variables = ('evis', 'ctheta'),
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
            xorders   = 3,
            yorder   = 5,
            provides = [ 'evis', 'ctheta', 'evis_edges' ],
            ),
        ibd_xsec = NestedDict(
            bundle = 'xsec_ibd_v01',
            order = 1,
            provides = [ 'ibd_xsec', 'ee', 'enu', 'jacobian' ]
            ),
        oscprob = NestedDict(
            bundle = 'oscprob_v01',
            name = 'oscprob',
            provides = ['oscprob', 'pmns']
            ),
        anuspec = NestedDict(
            bundle = 'reactor_anu_spectra_v02',
            name = 'anuspec',
            filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
            # strategy = dict( underflow='constant', overflow='extrapolate' ),
            edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
            ),
        eff = NestedDict(
            bundle = 'efficiencies_v01',
            correlated   = False,
            uncorrelated = True,
            norm         = True,
            names = dict(
                norm = 'global_norm'
                ),
            provides = [ 'eff', 'effunc_corr', 'effunc_uncorr', 'global_norm' ],
            efficiencies = 'data/dayabay/efficiency/P15A_efficiency.py'
            ),
        livetime = NestedDict(
            bundle = 'dayabay_livetime_hdf_v01',
            file   = 'data/dayabay/data/P15A/dubna/dayabay_data_dubna_v15_bcw_adsimple.hdf5',
            provides = ['livetime_daily', 'efflivetime_daily']
            ),
        baselines = NestedDict(
            bundle = 'baselines_v01',
            reactors  = 'data/dayabay/reactor/coordinates/coordinates_docDB_9757.py',
            detectors = 'data/dayabay/ad/coordinates/coordinates_docDB_9757.py',
            provides = [ 'baseline', 'baselineweight' ]
            ),
        thermal_power = NestedDict(
                bundle = 'dayabay_reactor_burning_info_v01',
                reactor_info = 'data/dayabay/reactor/power/WeeklyAvg_P15A_v1.txt.npz',
                provides = ['thermal_power', 'fission_fractions']
                ),
        iav = NestedDict(
                bundle     = 'detector_iav_db_root_v02',
                parname    = 'OffdiagScale',
                scale      = uncertain(1.0, 4, 'percent'),
                ndiag      = 1,
                filename   = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
                matrixname = 'iav_matrix'
                ),
        eres = NestedDict(
                bundle = 'detector_eres_common3_v02',
                # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                pars = uncertaindict(
                    [('Eres_a', 0.014764) ,
                        ('Eres_b', 0.0869) ,
                        ('Eres_c', 0.0271)],
                    mode='percent',
                    uncertainty=30
                    ),
                provides = [ 'eres', 'eres_matrix' ],
                expose_matrix = True
                )
        )

#
# Put the expression into context
context = ExpressionContext(cfg, ns=env.globalns)
a.build(context)

# Print the list of outputs, stored for the expression
from gna.bindings import OutputDescriptor
env.globalns.materializeexpressions(True)
env.globalns.printparameters( labels=True )
print( 'outputs:' )
print( context.outputs )

#
# Do some plots
#
# Initialize figure
if args.show:
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    # ax.set_xlabel()
    # ax.set_ylabel()
    # ax.set_title()

    out = context.outputs.result.AD11
    out.plot_hist()

if args.show:
    P.show()

#
# Dump the histogram to a dot graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot(context.outputs.ee, joints=False)
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

