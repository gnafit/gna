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
parser.add_argument( '-o', '--output', help='output figure name' )
parser.add_argument('--stats', action='store_true', help='show statistics')
parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
parser.add_argument('-e', '--embed', action='store_true', help='embed')
parser.add_argument('-i', '--indices', default='small', choices=['complete', 'small', 'minimal'], help='Set the indices coverage')
parser.add_argument('-m', '--mode', default='simple', choices=['simple', 'dyboscar', 'mid'], help='Set the topology')
parser.add_argument('-t', '--title', default='', help='figure title')
args = parser.parse_args()

#
# Import libraries
#
from gna.expression import *
from gna.configurator import uncertaindict, uncertain
from gna.bundle import execute_bundles
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
if args.indices=='complete':
    indices = [
        ('s', 'site',        ['EH1', 'EH2', 'EH3']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
        ('r', 'reactor',     ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']),
        ('i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.indices=='minimal':
    indices = [
        ('s', 'site',        ['EH1']),
        ('d', 'detector',    ['AD11'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11',))]))),
        ('r', 'reactor',     ['DB1']),
        ('i', 'isotope', ['U235']),
        ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
elif args.indices=='small':
    indices = [
        ('s', 'site',        ['EH1', 'EH2']),
        ('d', 'detector',    ['AD11', 'AD12', 'AD21'],
                             dict(short='s', name='site', map=OrderedDict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21',)) ]))),
        'name',
        ('r', 'reactor',     ['DB1', 'LA1']),
        ('i', 'isotope', ['U235']),
        ('c', 'component',   ['comp0', 'comp12']),
        ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
else:
    raise Exception('Unsupported indices '+args.indices)

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
        # cspec_diff_reac         = dict(expr='sum:i'),
        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
        # cspec_diff_det          = dict(expr='sum:r'),
        # spec_diff_det           = dict(expr='sum:c'),
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

        iso_spectrum_w          = dict(expr='kinint2*power_livetime_factor'),
        reac_spectrum           = dict(expr='sum:i|iso_spectrum_w'),
        reac_spectrum_w         = dict(expr='baselineweight*reac_spectrum'),
        ad_spectrum_c           = dict(expr='sum:r|reac_spectrum_w'),
        ad_spectrum_cw          = dict(expr='pmns*ad_spectrum_c'),
        ad_spectrum_w           = dict(expr='sum:c|ad_spectrum_cw'),

        # Accidentals
        acc_num_bf        = dict(expr='acc_norm*bkg_rate_acc*efflivetime',             label='Acc num {detector}\n(best fit)}'),
        bkg_acc           = dict(expr='acc_num_bf*bkg_spectrum_acc',                   label='Acc {detector}\n(w: {weight_label})'),

        # Li/He
        bkg_spectrum_li_w = dict(expr='bkg_spectrum_li*frac_li',                       label='9Li spectrum\n(frac)'),
        bkg_spectrum_he_w = dict(expr='bkg_spectrum_he*frac_he',                       label='8He spectrum\n(frac)'),
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

        # Total background
        bkg               = dict(expr='bkg_acc+bkg_alphan+bkg_amc+bkg_fastn+bkg_lihe', label='Background spectrum\n{detector}'),

        # dybOscar mode
        eres_cw           = dict(expr='eres*pmns'),
        # eres_w            = dict(expr='sum:c|eres_cw'),
        )

expr =[
        'baseline[d,r]',
        'enu| ee(evis()), ctheta()',
        'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
        'livetime=accumulate("livetime", livetime_daily[d]())',
        'power_livetime_factor_daily = efflivetime_daily[d]()*thermal_power[r]()*fission_fractions[i,r]()',
        'power_livetime_factor=accumulate("power_livetime_factor", power_livetime_factor_daily)',
        # Detector effects
        'eres_matrix| evis_hist()',
        'lsnl_edges| evis_edges(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
        # Bkg
        'bkg_acc = efflivetime * acc_norm[d] * bkg_rate_acc[d] * bkg_spectrum_acc[d]()',
        'bkg_lihe  = efflivetime[d] * bkg_rate_lihe[s]  * bracket| frac_li * bkg_spectrum_li() + frac_he * bkg_spectrum_he()',
        'fastn_shape[s]',
        'bkg_fastn = efflivetime[d] * bkg_rate_fastn[s] * bkg_spectrum_fastn[s]()',
        'bkg_amc   = efflivetime[d] * bkg_rate_amc[d] * bkg_spectrum_amc()',
        'bkg_alphan   = efflivetime[d] * bkg_rate_alphan[d] * bkg_spectrum_alphan[d]()',
        'bkg = bracket| bkg_acc + bkg_lihe + bkg_fastn + bkg_amc + bkg_alphan'
]

if args.mode=='dyboscar':
    expr.append(
        '''ibd =
                 global_norm*
                 eff*
                 effunc_uncorr[d]*
                 sum[c]|
                   pmns[c]*
                   eres[d]|
                     lsnl[d]|
                       iav[d]|
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
elif args.mode=='mid':
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
elif args.mode=='simple':
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
else:
    raise Exception('unsupported mode '+args.mode)

expr.append( 'observation=rebin| ibd + bkg' )
expr.append( 'total=concat[d]| observation' )

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
            xorders   = 2,
            yorder   = 2,
            provides = [ 'evis', 'ctheta', 'evis_edges', 'evis_hist' ],
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
                    [('eres.a', 0.014764) ,
                     ('eres.b', 0.0869) ,
                     ('eres.c', 0.0271)],
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
                names = [ 'li', 'he' ],
                format = 'frac_{component}',
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
parstats=dict()
env.globalns.printparameters(labels=True, stats=parstats)
print('Parameters stats:', parstats)
print( 'outputs:' )
if 'outputs' in args.print:
    print( context.outputs )
print(list( context.outputs.keys() ))

if 'inputs' in args.print:
    print( context.inputs )
print(list( context.inputs.keys() ))

if args.embed:
    import IPython
    IPython.embed()

if args.stats:
    from gna.graph import *
    out=context.outputs.concat_total
    walker = GraphWalker(out, context.outputs.thermal_power.DB1)
    report(out.data, fmt='Initial execution time: {total} s')
    report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
    print('Statistics', walker.get_stats())

    # times = walker.get_times(100)

#
# Do some plots
#
# Initialize figure
if args.show or args.output:
    from mpl_tools.helpers import savefig
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( L.u('evis') )
    ax.set_ylabel( 'Arbitrary units' )
    ax.set_title(args.title)

    def step(suffix):
        ax.legend(loc='upper right')
        savefig(args.output, suffix=suffix)

    outputs = context.outputs
    # outputs.kinint2.AD11.plot_hist(label='True spectrum')
    # step('_01_true')
    # outputs.iav.AD11.plot_hist(label='+IAV')
    # step('_02_iav')
    # outputs.lsnl.AD11.plot_hist(label='+LSNL')
    # step('_03_lsnl')
    # outputs.eres.AD11.plot_hist(label='+eres')
    if args.mode=='dyboscar':
        out = outputs.sum_sum_sum_eres_cw
    else:
        out = outputs.eres
    out.AD11.plot_hist(label='EH1 AD1')
    step('_04_eres')


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

