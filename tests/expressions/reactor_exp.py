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
from gna.configurator import uncertaindict
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
        ('r', 'reactor',     ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA3']),
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

#
# Intermediate options (empty for now)
#
lib = dict(
        cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob'),
        cspec_diff_reac         = dict(expr='sum:i'),
        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
        cspec_diff_det          = dict(expr='sum:r'),
        spec_diff_det           = dict(expr='sum:c'),
        cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),
        norm_bf                 = dict(expr='eff*efflivetime*effunc_uncorr*global_norm')
        )

expr =[
        'baseline[d,r]',
        'enu| ee(evis()), ctheta()',
        'jacobian| enu(), ee(), ctheta()',
        'ibd_xsec(enu(), ctheta())',
        'oscprob[c,d,r]( enu() )',
        'anuspec[i](enu())',
        '''result = global_norm *  eff * efflivetime[d] * effunc_uncorr[d] *
                      sum[c]| pmns[c]*
                        sum[r]|
                          baselineweight[r,d]*
                          sum[i]|
                            kinint2|
                              anuspec() * oscprob() * ibd_xsec() * jacobian()'''
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
            provides = [ 'evis', 'ctheta' ],
            ),
        ibd_xsec = NestedDict(
            bundle = 'xsec_ibd_v01',
            order = 1,
            provides = [ 'ibd_xsec', 'ee', 'enu', 'jacobian' ]
            ),
        oscprob = NestedDict(
            bundle = 'oscprob_v01',
            name = 'oscprob',
            input = True,
            size = 10,
            debug = False,
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
            provides = ['livetime', 'efflivetime']
            ),
        baselines = NestedDict(
            bundle = 'baselines',
            detectors = OrderedDict([
                ('AD11', [ 362.8329,    50.4206, -70.8174 ]),
                ('AD12', [ 358.8044,    54.8583, -70.8135 ]),
                ('AD21', [   7.6518,  -873.4882, -67.5241 ]),
                ('AD22', [   9.5968,  -879.149,  -67.5202 ]),
                ('AD31', [ 936.7486, -1419.013,  -66.4852 ]),
                ('AD32', [ 941.4493, -1415.305,  -66.4966 ]),
                ('AD33', [ 940.4612, -1423.737,  -66.4965 ]),
                ('AD34', [ 945.1678, -1420.0282, -66.4851 ]),
                ]),
            reactors = OrderedDict([
                ('DB1', [  359.2029,  411.4896, -40.2308 ]),
                ('DB2', [  448.0015,  411.0017, -40.2392 ]),
                ('LA1', [ -319.666,  -540.7481, -39.7296 ]),
                ('LA2', [ -267.0633, -469.2051, -39.7230 ]),
                ('LA3', [ -543.284,  -954.7018, -39.7987 ]),
                ('LA4', [ -490.6906, -883.152,  -39.7884 ]),
                ]),
            provides = [ 'baseline', 'baselineweight' ]
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

