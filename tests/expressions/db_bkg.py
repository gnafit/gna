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
from collections import OrderedDict
R.GNAObject

#
# Define the indices (empty for current example)
#
if args.mode=='complete':
    indices = [
        ('i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']),
        ('r', 'reactor',     ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
        ('s', 'site',        ['EH1', 'EH2', 'EH3']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.mode=='minimal':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1']),
        ('d', 'detector',    ['AD11'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12'))]))),
        ('s', 'site',        ['EH1']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.mode=='small':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1', 'LA1']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')) ]))),
        ('s', 'site',        ['EH1', 'EH2']),
        ('c', 'component',   ['comp0', 'comp12']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
else:
    raise Exception('Unsupported mode '+args.mode)

detectors = ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']
groups=NestedDict(
        exp  = { 'dayabay': detectors },
        det  = { d: (d,) for d in detectors },
        site = NestedDict([
            ('EH1', ['AD11', 'AD12']),
            ('EH2', ['AD21', 'AD22']),
            ('EH3', ['AD31', 'AD32', 'AD33', 'AD34']),
            ]),
        adnum_local = NestedDict([
            ('1', ['AD11', 'AD21', 'AD31']),
            ('2', ['AD12', 'AD22', 'AD32']),
            ('3', ['AD33']),
            ('4', ['AD34']),
            ]),
        adnum_global = NestedDict([
            ('1', ['AD11']), ('2', ['AD12']),
            ('3', ['AD21']), ('8', ['AD22']),
            ('4', ['AD31']), ('5', ['AD32']), ('6', ['AD33']), ('7', ['AD34']),
            ]),
        adnum_global_alphan_subst = NestedDict([
            ('1', ['AD11']), ('2', ['AD12']),
            ('3', ['AD21', 'AD22']),
            ('4', ['AD31']), ('5', ['AD32']), ('6', ['AD33', 'AD34']),
            ])
        )

lib = OrderedDict(
        # Accidentals
        acc_num_bf        = dict(expr='acc_norm*bkg_rate_acc*efflivetime',             label='Acc num {detector}\n(best fit)}'),
        bkg_acc           = dict(expr='acc_num_bf*bkg_spectrum_acc',                   label='Acc {detector}\n(w: {weight_label})'),

        # Li/He
        bkg_spectrum_li_w = dict(expr='bkg_spectrum_li*frac_li',        label='9Li spectrum\n(frac)'),
        bkg_spectrum_he_w = dict(expr='bkg_spectrum_he*frac_he',        label='8He spectrum\n(frac)'),
        bkg_spectrum_lihe = dict(expr='bkg_spectrum_he_w+bkg_spectrum_li_w',           label='8He/9Li spectrum\n(norm)'),
        lihe_num_bf       = dict(expr='bkg_rate_lihe*efflivetime'),
        bkg_lihe          = dict(expr='bkg_spectrum_lihe*lihe_num_bf',                 label='8He/9Li {detector}\n(w: {weight_label})'),

        # Fast neutrons
        fastn_num_bf      = dict(expr='bkg_rate_fastn*efflivetime'),
        bkg_fastn         = dict(expr='bkg_spectrum_fastn*fastn_num_bf',               label='Fast neutron {detector}\n(w: {weight_label})'),

        # AmC
        amc_num_bf        = dict(expr='bkg_rate_amc*efflivetime'),
        bkg_amc           = dict(expr='bkg_spectrum_amc*amc_num_bf',                   label='AmC {detector}\n(w: {weight_label})'),

        # AlphaN
        alphan_num_bf     = dict(expr='bkg_rate_alphan*efflivetime'),
        bkg_alphan        = dict(expr='bkg_spectrum_alphan*alphan_num_bf',             label='C(alpha,n) {detector}\n(w: {weight_label})'),

        # Total
        bkg               = dict(expr='bkg_acc+bkg_alphan+bkg_amc+bkg_fastn+bkg_lihe', label='Background spectrum\n{detector}')
        )

expr =[
        'evis_edges',
        'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
        'livetime=accumulate("livetime", livetime_daily[d]())',
        'bkg_acc = efflivetime * acc_norm[d] * bkg_rate_acc[d] * bkg_spectrum_acc[d]()',
        'bkg_lihe  = efflivetime[d] * bkg_rate_lihe[s]  * bracket| frac_li * bkg_spectrum_li() + frac_he * bkg_spectrum_he()',
        'fastn_shape[s]',
        'bkg_fastn = efflivetime[d] * bkg_rate_fastn[s] * bkg_spectrum_fastn[s]()',
        'bkg_amc   = efflivetime[d] * bkg_rate_amc[d] * bkg_spectrum_amc()',
        'bkg_alphan   = efflivetime[d] * bkg_rate_alphan[d] * bkg_spectrum_alphan[d]()',
        'bkg = bracket| bkg_acc + bkg_lihe + bkg_fastn + bkg_amc + bkg_alphan',
        'common = concat[d]| bkg'
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
        #
        # Spectra
        #
        bkg_spectrum_acc = NestedDict(
            bundle    = 'root_histograms_v02',
            filename  = 'data/dayabay/data_spectra/P15A_IHEP_data/P15A_All_raw_sepctrum_coarse.root',
            format    = '{site}_AD{adnum_local}_singleTrigEnergy',
            name      = 'bkg_spectrum_acc',
            label     = 'Accidentals {detector}\n(norm spectrum)',
            groups    = groups,
            normalize = True,
            ),
        bkg_spectrum_li=NestedDict(
            bundle    = 'root_histograms_v02',
            filename  = 'data/dayabay/bkg/lihe/toyli9spec_BCWmodel_v1.root',
            format    = 'h_eVisAllSmeared',
            name      = 'bkg_spectrum_li',
            label     = '9Li spectrum\n(norm)',
            normalize = True,
            ),
        bkg_spectrum_he= NestedDict(
            bundle    = 'root_histograms_v02',
            filename  = 'data/dayabay/bkg/lihe/toyhe8spec_BCWmodel_v1.root',
            format    = 'h_eVisAllSmeared',
            name      = 'bkg_spectrum_he',
            label     = '8He spectrum\n(norm)',
            normalize = True,
            ),
        bkg_spectrum_amc = NestedDict(
            bundle    = 'root_histograms_v02',
            filename  = 'data/dayabay/bkg/P12B_amc_expofit.root',
            format    = 'hCorrAmCPromptSpec',
            name      = 'bkg_spectrum_amc',
            label     = 'AmC spectrum\n(norm)',
            normalize = True,
            ),
        bkg_spectrum_alphan = NestedDict(
            bundle    = 'root_histograms_v02',
            filename  = 'data/dayabay/bkg/P12B_alphan_coarse.root',
            format    = 'AD{adnum_global_alphan_subst}',
            groups    = groups,
            name      = 'bkg_spectrum_alphan',
            label     = 'C(alpha,n) spectrum\n{detector} (norm)',
            normalize = True,
            ),
        lihe_fractions=NestedDict(
                bundle = 'var_fractions_v01',
                format = 'frac_{component}',
                names = [ 'li', 'he' ],
                fractions = uncertaindict(
                    li = ( 0.95, 0.05, 'relative' )
                    ),
                provides = [ 'frac_li', 'frac_he' ]
                ),
        bkg_spectrum_fastn=NestedDict(
                bundle='dayabay_fastn_v02',
                parameter='fastn_shape',
                name='bkg_spectrum_fastn',
                normalize=(0.7, 12.0),
                bins='evis_edges',
                order=2,
                ),
        #
        # Parameters
        #
        fastn_shape=NestedDict(
                bundle='parameters_1d_v01',
                parameter='fastn_shape',
                label='Fast neutron shape parameter for {site}',
                pars=uncertaindict(
                    [ ('EH1', (67.79, 0.1132)),
                        ('EH2', (58.30, 0.0817)),
                        ('EH3', (68.02, 0.0997)) ],
                    mode='relative',
                    ),
                ),
        #
        # Livetime
        #
        livetime = NestedDict(
                bundle = 'dayabay_livetime_hdf_v01',
                file   = 'data/dayabay/data/P15A/dubna/dayabay_data_dubna_v15_bcw_adsimple.hdf5',
                provides = ['livetime_daily', 'efflivetime_daily']
                ),
        #
        # Rates
        #
        bkg_rate_acc = NestedDict(
                bundle    ='parameters_1d_v01',
                parameter = 'bkg_rate_acc',
                label='Acc rate at {detector}',
                pars = uncertaindict(
                    [
                        ( 'AD11', 8.463602 ),
                        ( 'AD12', 8.464611 ),
                        ( 'AD21', 6.290756 ),
                        ( 'AD22', 6.180896 ),
                        ( 'AD31', 1.273798 ),
                        ( 'AD32', 1.189542 ),
                        ( 'AD33', 1.197807 ),
                        ( 'AD34', 0.983096 ),
                        ],
                    uncertainty = 1.0,
                    mode = 'percent',
                    ),
                separate_uncertainty='acc_norm',
                provides = ['acc_norm']
                ),
        bkg_rate_lihe = NestedDict(
                bundle    ='parameters_1d_v01',
                parameter = 'bkg_rate_lihe',
                label='⁹Li/⁸He rate at {site}',
                pars = uncertaindict([
                    ('EH1', (2.46, 1.06)),
                    ('EH2', (1.72, 0.77)),
                    ('EH3', (0.15, 0.06))],
                    mode = 'absolute',
                    ),
                ),
        bkg_rate_fastn = NestedDict(
                bundle    ='parameters_1d_v01',
                parameter = 'bkg_rate_fastn',
                label='Fast neutron rate at {site}',
                pars = uncertaindict([
                    ('EH1', (0.792, 0.103)),
                    ('EH2', (0.566, 0.074)),
                    ('EH3', (0.047, 0.009))],
                    mode = 'absolute',
                    ),
                ),
        bkg_rate_amc = NestedDict(
                bundle    ='parameters_1d_v01',
                parameter = 'bkg_rate_amc',
                label='AmC rate at {detector}',
                pars = uncertaindict(
                    [
                        ('AD11', (0.18, 0.08)),
                        ('AD12', (0.18, 0.08)),
                        ('AD21', (0.16, 0.07)),
                        ('AD22', (0.15, 0.07)),
                        ('AD31', (0.07, 0.03)),
                        ('AD32', (0.06, 0.03)),
                        ('AD33', (0.07, 0.03)),
                        ('AD34', (0.05, 0.02)),
                        ],
                    mode = 'absolute',
                    ),
                ),
        bkg_rate_alphan = NestedDict(
                bundle    ='parameters_1d_v01',
                parameter = 'bkg_rate_alphan',
                label='C(alpha,n) rate at {detector}',
                pars = uncertaindict([
                        ('AD11', (0.08)),
                        ('AD12', (0.07)),
                        ('AD21', (0.05)),
                        ('AD22', (0.07)),
                        ('AD31', (0.05)),
                        ('AD32', (0.05)),
                        ('AD33', (0.05)),
                        ('AD34', (0.05)),
                        ],
                        uncertainty = 50,
                        mode = 'percent',
                    ),
                ),
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

    # out = context.outputs.observation.AD11
    # out.plot_hist()

if args.show:
    P.show()

#
# Dump the histogram to a dot graph
#
if args.dot:
    try:
        from gna.graphviz import GNADot

        graph = GNADot(context.outputs.bkg.AD11, joints=False)
        graph.write(args.dot)
        print( 'Write output to:', args.dot )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise


