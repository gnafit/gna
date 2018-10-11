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
        ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.mode=='minimal':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1']),
        ('d', 'detector',    ['AD11']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23'])
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.mode=='small':
    indices = [
        ('i', 'isotope', ['U235']),
        ('r', 'reactor',     ['DB1', 'LA1']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21']),
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
        cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                       label='anu count rate\n{isotope}@{reactor}->{detector} ({component})'),
        cspec_diff_reac         = dict(expr='sum:i'),
        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
        cspec_diff_det          = dict(expr='sum:r'),
        spec_diff_det           = dict(expr='sum:c'),
        cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),

        norm_bf                 = dict(expr='eff*effunc_uncorr*global_norm'),
        ibd                     = dict(expr='eres*norm_bf', label='Observed IBD spectrum\n{detector}'),

        lsnl_component_weighted = dict(expr='lsnl_component*lsnl_weight'),
        lsnl_correlated         = dict(expr='sum:l|lsnl_component_weighted'),
        evis_nonlinear_correlated = dict(expr='evis_edges*lsnl_correlated'),
        evis_nonlinear          = dict(expr='escale*evis_nonlinear_correlated'),

        oscprob_weighted        = dict(expr='oscprob*pmns'),
        oscprob_full            = dict(expr='sum:c|oscprob_weighted', label='anue survival probability\nweight: {weight_label}'),

        anuspec_weighted        = dict(expr='anuspec*power_livetime_factor'),
        anuspec_rd              = dict(expr='sum:i|anuspec_weighted', label='anue spectrum {reactor}->{detector}\nweight: {weight_label}'),

        countrate_rd            = dict(expr='anuspec_rd*ibd_xsec*jacobian*oscprob_full'),
        countrate_weighted      = dict(expr='baselineweight*countrate_rd'),
        countrate               = dict(expr='sum:r|countrate_weighted', label='Count rate {detector}\nweight: {weight_label}'),

        observation_raw         = dict(expr='bkg+ibd', label='Observed spectrum\n{detector}'),

        bkg_spectrum_li_w       = dict(expr='bkg_spectrum_li*frac_li', label='9Li spectrum\n(frac)'),
        bkg_spectrum_he_w       = dict(expr='bkg_spectrum_he*frac_he', label='8He spectrum\n(frac)'),
        bkg_spectrum_lihe       = dict(expr='bkg_spectrum_he_w+bkg_spectrum_li_w', label='8He/9Li spectrum\n(norm)'),
        bkg                     = dict(expr='bkg_spectrum_acc+bkg_spectrum_lihe', label='Background spectrum\n{detector}')
        )

expr =[
        'baseline[d,r]',
        'enu| ee(evis()), ctheta()',
        'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
        'livetime=accumulate("livetime", livetime_daily[d]())',
        'power_livetime_factor_daily = efflivetime_daily[d]()*thermal_power[r]()*fission_fractions[i,r]()',
        'power_livetime_factor=accumulate("power_livetime_factor", power_livetime_factor_daily)',
        'eres_matrix| evis_edges()',
        'lsnl_edges| evis_edges(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
        'bkg_spectrum_lihe = bracket| frac_li * bkg_spectrum_li() + frac_he * bkg_spectrum_he()',
        'bkg = bracket| bkg_spectrum_acc[d]()+bkg_spectrum_lihe'
]

if False:
    expr.append(
        '''ibd =
                 global_norm*
                 eff*
                 effunc_uncorr[d]*
                 eres[d]|
                   lsnl[d]|
                     iav[d]|
                       sum[c]|
                         pmns[c]*
                         sum[r]|
                           baselineweight[r,d]*
                           sum[i]|
                             power_livetime_factor*
                             kinint2|
                               anuspec[i](enu())*
                               oscprob[c,d,r](enu())*
                               ibd_xsec(enu(), ctheta())*
                               jacobian(enu(), ee(), ctheta())
        ''')
elif True:
    expr.append('''ibd =
                      global_norm*
                      eff*
                      effunc_uncorr[d]*
                      eres[d]|
                        lsnl[d]|
                          iav[d]|
                              kinint2|
                                sum[r]|
                                  baselineweight[r,d]*
                                  ibd_xsec(enu(), ctheta())*
                                  jacobian(enu(), ee(), ctheta())*
                                  (sum[i]| power_livetime_factor*anuspec[i](enu()))*
                                  sum[c]|
                                    pmns[c]*oscprob[c,d,r](enu())
        ''')

expr.append( 'observation=rebin| ibd + bkg' )

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
            provides = [ 'baseline', 'baselineweight' ],
            units = 'meters'
            ),
        thermal_power = NestedDict(
                bundle = 'dayabay_reactor_burning_info_v01',
                reactor_info = 'data/dayabay/reactor/power/WeeklyAvg_P15A_v1.txt.npz',
                fission_uncertainty_info = 'data/dayabay/reactor/fission_fraction/2013.12.05_xubo.py',
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
                ),
        lsnl = NestedDict(
                bundle     = 'detector_nonlinearity_db_root_v02',
                names      = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                filename   = 'data/dayabay/tmp/detector_nl_consModel_450itr.root',
                parnames      = dict(
                    lsnl   = 'lsnl_weight',
                    escale = 'escale'
                    ),
                par        = uncertain(1.0, 0.2, 'percent'),
                edges      = 'evis_edges',
                provides   = ['lsnl', 'lsnl_component', 'escale', 'lsnl_weight', 'lsnl_edges']
                ),
        rebin = NestedDict(
                bundle = 'rebin_v02',
                rounding = 3,
                edges = N.concatenate(( [0.7], N.arange(1.2, 8.1, 0.2), [12.0] ))
                ),
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

    out = context.outputs.observation.AD11
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

        graph = GNADot(context.outputs.thermal_power.values(), joints=False)
        name = args.dot.replace('.dot', '_lt.dot')
        graph.write(name)
        print( 'Write output to:', name )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

