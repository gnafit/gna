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
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
        ('s', 'site',        ['EH1']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.mode=='small':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1', 'LA1']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
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
        bkg_spectrum_li_w       = dict(expr='bkg_spectrum_li*frac_li', label='9Li spectrum\n(frac)'),
        bkg_spectrum_he_w       = dict(expr='bkg_spectrum_he*frac_he', label='8He spectrum\n(frac)'),
        bkg_spectrum_lihe       = dict(expr='bkg_spectrum_he_w+bkg_spectrum_li_w', label='8He/9Li spectrum\n(norm)'),
        bkg                     = dict(expr='bkg_spectrum_acc+bkg_spectrum_fastn+bkg_spectrum_lihe', label='Background spectrum\n{detector}')
        )

expr =[
        'bkg_spectrum_fastn[s]()',
        # 'bkg = bracket| bkg_spectrum_acc[d]()+bkg_spectrum_lihe+bkg_spectrum_fastn[s]()'
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
        lihe_fractions=NestedDict(
                bundle = 'var_fractions_v01',
                names = [ 'li', 'he' ],
                fractions = uncertaindict(
                    li = ( 0.95, 0.05, 'relative' )
                    ),
                provides = [ 'frac_li', 'frac_he' ]
                ),
        bkg_spectrum_fastn=NestedDict(
            bundle='dayabay_fastn_v02',
            parameter='fastn_shape',
            normalize=(0.7, 12.0),
            bins=N.linspace(0.0, 12.0, 241),
            order=2,
            pars=uncertaindict(
               [ ('EH1', (67.79, 0.1132)),
                 ('EH2', (58.30, 0.0817)),
                 ('EH3', (68.02, 0.0997)) ],
                mode='relative',
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
# if args.dot:
    # try:
        # from gna.graphviz import GNADot

        # graph = GNADot(context.outputs.ee, joints=False)
        # graph.write(args.dot)
        # print( 'Write output to:', args.dot )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
        # raise


