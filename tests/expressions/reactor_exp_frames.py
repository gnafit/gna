#!/usr/bin/env python

#
# Initialize argument parser
#
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '--dot', help='write graphviz output', required=True )
parser.add_argument('-m', '--mode', default='small', choices=['complete', 'small', 'minimal'], help='Set the indices coverage')
parser.add_argument('-s', '--stage', type=int, help='stage', required=True)
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
R.GNAObject

#
# Define the indices (empty for current example)
#
if args.mode=='complete':
    indices = [
        ('i', 'isotope', ['U235', 'U238', 'Pu239', 'Pu241']),
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

lib = dict(
        cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                       label='anu count rate\n{isotope}@{reactor}->{detector} ({component})'),
        cspec_diff_reac         = dict(expr='sum:i'),
        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
        cspec_diff_det          = dict(expr='sum:r'),
        spec_diff_det           = dict(expr='sum:c'),
        cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),
        # norm_bf                 = dict(expr='eff*efflivetime*effunc_uncorr*global_norm'),
        norm_bf                 = dict(expr='eff*effunc_uncorr*global_norm'),
        observation             = dict(expr='eres*norm_bf', label='Observed spectrum\n{detector}'),
        lsnl_component_weighted = dict(expr='lsnl_component*lsnl_weight'),
        lsnl_correlated         = dict(expr='sum:l'),
        evis_nonlinear_correlated = dict(expr='evis_edges*lsnl_correlated'),
        evis_nonlinear          = dict(expr='escale*evis_nonlinear_correlated'),
        )

stages = [
        ( [ 'enu| ee(evis()), ctheta()' ],
          'enu',
          'main'),
        ( [
          'power_livetime_factor_daily = efflivetime_daily[d]()*thermal_power[r]()*fission_fractions[i,r]()',
          ],
          'efflivetime_daily',
          'power'),
        ( [
          'eres_matrix| evis_edges()',
          'eres[d]'
          ],
          'evis_edges',
          'evis'),
        ( [
          'evis()',
          'lsnl_edges| evis_edges(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
          'lsnl[d]'
          ],
          'evis_edges',
          'lsnl'),
        ( [ 'iav[d]', ],
          'iav',
          'iav'),
        ( [
          'enu| ee(evis()), ctheta()',
          'baseline[d,r]',
          '''result = anuspec[i](enu())*
                      oscprob[c,d,r](enu())*
                      ibd_xsec(enu(), ctheta())*
                      jacobian(enu(), ee(), ctheta())'''
                      ],
          'enu',
          'crate'),
        ( [
          'baseline[d,r]',
          'enu| ee(evis()), ctheta()',
          '''result = kinint2| anuspec[i](enu())*
                      oscprob[c,d,r](enu())*
                      ibd_xsec(enu(), ctheta())*
                      jacobian(enu(), ee(), ctheta())'''
                      ],
          'enu',
          'crate_int'),
        ( [
          'baseline[d,r]',
          'enu| ee(evis()), ctheta()',
          'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
          'livetime=accumulate("livetime", livetime_daily[d]())',
          'power_livetime_factor_daily = efflivetime_daily[d]()*thermal_power[r]()*fission_fractions[i,r]()',
          'power_livetime_factor=accumulate("power_livetime_factor", power_livetime_factor_daily)',
          'eres_matrix| evis_edges()',
          'lsnl_edges| evis_edges(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
          '''result = rebin|
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
                                      jacobian(enu(), ee(), ctheta())'''
                                      ],
          'enu',
          'main'),
        ]

def build_and_plot(expr, obj, suffix):
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
                units="meters"
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
                    )
            )
    #
    # Put the expression into context
    context = ExpressionContext(cfg, ns=env.globalns)
    a.build(context)

    # # Print the list of outputs, stored for the expression
    # from gna.bindings import OutputDescriptor
    # env.globalns.materializeexpressions(True)
    # env.globalns.printparameters( labels=True )
    # print( 'outputs:' )
    # print( context.outputs )

    try:
        from gna.graphviz import GNADot

        obj = context.outputs[obj]
        if isinstance(obj, NestedDict):
            obj = obj.values()
        if len(expr)<4:
            label = expr[0]
            if label=='evis()':
                label=expr[1]
        else:
            label = ''
        graph = GNADot(obj, joints=False, label=label)
        name = args.dot.replace('.dot', '_'+suffix+'.dot')
        graph.write(name)
        print( 'Write output to:', name )
    except Exception as e:
        print( '\033[31mFailed to plot dot\033[0m' )
        raise

build_and_plot(*stages[args.stage])
