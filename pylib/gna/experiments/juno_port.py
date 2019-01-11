# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna.expression.index import NIndex
import numpy as np

seconds_per_day = 60*60*24
class exp(baseexp):
    detectorname = 'AD1'

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument( '-o', '--output', help='output figure name' )
        parser.add_argument('--stats', action='store_true', help='show statistics')
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-e', '--embed', action='store_true', help='embed')
        parser.add_argument('-c', '--composition', default='minimal', choices=['complete', 'minimal'], help='Set the indices coverage')
        parser.add_argument('-m', '--mode', default='simple', choices=['simple', 'dyboscar', 'mid'], help='Set the topology')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')

    def __init__(self, env, opts):
        baseexp.__init__(self, env, opts)

        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.build()
        self.register()

    def init_nidx(self):
        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] )
        ]
        if self.opts.composition=='minimal':
            self.nidx[1][2] = self.nidx[1][2][:1]
            self.nidx[2][2] = self.nidx[2][2][:1]

        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        self.formula = self.formula_base
        versions = dict(
                simple = self.formula_ibd_simple,
                dyboscar = self.formula_ibd_do,
                mid = self.formula_ibd_mid
                )
        self.formula.append(versions[self.opts.mode])
        self.formula+=self.formula_back

    def init_configuration(self):
        self.cfg = NestedDict(
                kinint2 = NestedDict(
                    bundle   = 'integral_2d1d_v01',
                    variables = ('evis', 'ctheta'),
                    edges    = np.arange(0.0, 12.001, 0.02),
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
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
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
                fission_fractions = NestedDict(
                    bundle = dict(name="parameters",
                                  version = "v01",
                                  nidx = self.nidx.get_subset(('r', 'i')),
                                  major = 'i'
                                  ),
                             parameter = "fission_fractions",
                             label = 'Fission fraction of {isotope} in reactor {reactor}',
                             pars = uncertaindict([
                                 ('U235',  0.60),
                                 ('Pu239', 0.27),
                                 ('U238',  0.07),
                                 ('Pu241', 0.06)
                                 ],
                                uncertainty = 30.0,
                                mode = 'percent',
                            ),
                      provides=['fission_fractions']
                    ),
                livetime = NestedDict(
                        bundle = dict(name="parameters",
                                      version = "v01",
                                      nidx = self.nidx.get_subset(('d',))),
                        parameter = "livetime",
                        label = 'Livetime of {detector} in seconds',
                        pars = uncertaindict(
                            [('AD1', (6*365*seconds_per_day, 'fixed'))],
                            ),
                        provides=['livetime']
                        ),
                efflivetime = NestedDict(
                        bundle = dict(name="parameters",
                                      version = "v01",
                                      nidx = self.nidx.get_subset(('d',))),
                        parameter = "efflivetime",
                        label = 'Effective livetime of {detector} in seconds',
                        pars = uncertaindict(
                            [('AD1', (6*365*seconds_per_day*0.8, 'fixed'))],
                            ),
                        provides=['efflivetime']
                        ),
                baselines = NestedDict(
                    bundle = dict(name='reactor_baselines', version='v01',
                                  nidx = self.nidx.get_subset('rd'),
                                  major = 'rd'
                        ),
                    reactors  = 'data/juno_nominal/coordinates_reactors.py',
                    detectors = 'data/juno_nominal/coordinates_det.py',
                    provides = [ 'baseline', 'baselineweight' ],
                    unit = 'km'
                    ),
                thermal_power = NestedDict(
                        bundle = dict(name="parameters",
                                      version = "v01",
                                      nidx = self.nidx.get_subset(('r',))),
                        parameter = "thermal_power",
                        label = 'Thermal power of {reactor} in MWt',
                        pars = uncertaindict([
                            ('TS1',  4.6),
                            ('TS2',  4.6),
                            ('TS3',  4.6),
                            ('TS4',  4.6),
                            ('YJ1',  2.9),
                            ('YJ2',  2.9),
                            ('YJ3',  2.9),
                            ('YJ4',  2.9),
                            ('YJ5',  2.9),
                            ('YJ6',  2.9),
                            ('DYB', 17.4),
                            ('HZ',  17.4),
                            ],
                            uncertainty=None,
                            mode='fixed'
                            ),
                        provides=["thermal_power"]
                        ),
                target_protons = NestedDict(
                        bundle = dict(name="parameters",
                                      version = "v01",
                                      nidx = self.nidx.get_subset(('d'))),
                        parameter = "target_protons",
                        label = 'Number of protons in {detector}',
                        pars = uncertaindict(
                            [('AD1', (1.42e33, 'fixed'))],
                            ),
                        provides=["target_protons"]
                        ),
                eper_fission =  NestedDict(
                        bundle = dict(name="parameters",
                                      version = "v01",
                                      nidx = self.nidx.get_subset(('i', ))),
                        parameter = "Eper_fission",
                        label = 'Energy per fission for {isotope} in MeV',
                        pars = uncertaindict(
                            [('Pu239', (209.99, 0.60)),
                             ('Pu241', (213.60, 0.65)),
                             ('U235',  (201.92, 0.46)),
                             ('U238', (205.52, 0.96))],
                            mode='absolute'
                            ),
                        provides=["eper_fission"]
                        ),
                eres = NestedDict(
                        bundle = 'detector_eres_common3_v02',
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        pars = uncertaindict(
                            [('eres.a', 0.001) ,
                             ('eres.b', 0.03) ,
                             ('eres.c', 0.001)],
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
                        edges = np.concatenate(( [0.7], np.arange(1, 8, 0.02), [12.0] ))
                        ),
                )

    def build(self):
        from gna.expression import Expression, ExpressionContext
        from gna.bundle import execute_bundles

        # Initialize the expression and indices
        self.expression = Expression(self.formula, self.nidx)

        # Dump the information
        if self.opts.verbose:
            print(self.expression.expressions_raw)
            print(self.expression.expressions)

        # Parse the expression
        self.expression.parse()
        # The next step is needed to name all the intermediate variables.
        self.expression.guessname(self.lib, save=True)

        if self.opts.verbose>1:
            print('Expression tree:')
            self.expression.tree.dump(True)
            print()

        # Put the expression into context
        self.context = ExpressionContext(self.cfg, ns=self.env.globalns)
        self.expression.build(self.context)

        if self.opts.verbose>1:
            width = 40
            print('Outputs:')
            print(self.context.outputs.__str__(nested=True, width=width))

            print()
            print('Inputs:')
            print(self.context.inputs.__str__(nested=True, width=width))

            print()

        if self.opts.verbose:
            print('Parameters:')
            self.env.globalns.printparameters(labels=True)

    def register(self):
        ns = self.env.globalns
        outputs = self.context.outputs
        # ns.addobservable("{0}_unoscillated".format(self.detectorname), outputs, export=False)
        ns.addobservable("{0}_noeffects".format(self.detectorname),    outputs.observation_noeffects.AD1, export=False)
        ns.addobservable("{0}_fine".format(self.detectorname),         outputs.ibd.AD1)
        ns.addobservable("{0}".format(self.detectorname),              outputs.rebin.AD1)

    formula_base = [
            'baseline[d,r]',
            'enu| ee(evis()), ctheta()',
            'livetime[d]',
            'efflivetime[d]',
            'eper_fission[i]',
            'power_livetime_factor =  efflivetime[d] * thermal_power[r] * fission_fractions[r,i]',
            # Detector effects
            'eres_matrix| evis_hist()',
            'lsnl_edges| evis_hist(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
            'norm_bf = global_norm* eff* effunc_uncorr[d]'
    ]

    formula_ibd_do = '''ibd =
                     norm_bf[d]*
                     sum[c]|
                       pmns[c]*
                       eres[d]|
                         lsnl[d]|
                             sum[r]|
                               baselineweight[r,d]*
                               sum[i]|
                                 power_livetime_factor*
                                 kinint2|
                                   anuspec[i](enu())*
                                   oscprob[c,d,r](enu())*
                                   ibd_xsec(enu(), ctheta())*
                                   jacobian(enu(), ee(), ctheta())
            '''

    formula_ibd_mid = '''ibd =
                     norm_bf[d]*
                     eres[d]|
                       lsnl[d]|
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
            '''

    formula_ibd_simple = '''ibd =
                            norm_bf[d]*
                            eres[d]|
                              lsnl[d]|
                                    kinint2|
                                      sum[r]|
                                        baselineweight[r,d]*
                                        ibd_xsec(enu(), ctheta())*
                                        jacobian(enu(), ee(), ctheta())*
                                        (sum[i]|  power_livetime_factor*anuspec[i](enu()))*
                                        sum[c]|
                                          pmns[c]*oscprob[c,d,r](enu())
            '''

    formula_back = [
            'observation_noeffects = norm_bf[d]*kinint2[d]()',
            'observation=rebin| ibd'
            ]

    lib = dict(
            cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                           label='anu count rate\n{isotope}@{reactor}->{detector} ({component})'),
            cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
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

            eres_cw           = dict(expr='eres*pmns'),
            )

# #
# # Import libraries
# #
# from load import ROOT as R
# from gna.env import env
# from matplotlib import pyplot as P
# from mpl_tools import bindings
# from gna.labelfmt import formatter as L
# from collections import OrderedDict
# R.GNAObject





# # Print the list of outputs, stored for the expression
# from gna.bindings import OutputDescriptor
# env.globalns.materializeexpressions(True)
# parstats=dict()
# env.globalns.printparameters(labels=True, stats=parstats)
# print('Parameters stats:', parstats)
# print( 'outputs:' )
# if 'outputs' in args.print:
    # print( context.outputs )
# print(list( context.outputs.keys() ))

# if 'inputs' in args.print:
    # print( context.inputs )
# print(list( context.inputs.keys() ))

# if args.embed:
    # import IPython
    # IPython.embed()

# if args.stats:
    # from gna.graph import *
    # out=context.outputs.concat_total
    # walker = GraphWalker(out, context.outputs.thermal_power.DB1)
    # report(out.data, fmt='Initial execution time: {total} s')
    # report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
    # print('Statistics', walker.get_stats())

    # # times = walker.get_times(100)

# #
# # Do some plots
# #
# # Initialize figure
# if args.show or args.output:
    # from mpl_tools.helpers import savefig
    # fig = P.figure()
    # ax = P.subplot( 111 )
    # ax.minorticks_on()
    # ax.grid()
    # ax.set_xlabel( L.u('evis') )
    # ax.set_ylabel( 'Arbitrary units' )
    # ax.set_title(args.title)

    # def step(suffix):
        # ax.legend(loc='upper right')
        # savefig(args.output, suffix=suffix)

    # outputs = context.outputs
    # # outputs.kinint2.AD11.plot_hist(label='True spectrum')
    # # step('_01_true')
    # # outputs.iav.AD11.plot_hist(label='+IAV')
    # # step('_02_iav')
    # # outputs.lsnl.AD11.plot_hist(label='+LSNL')
    # # step('_03_lsnl')
    # # outputs.eres.AD11.plot_hist(label='+eres')
    # if args.mode=='dyboscar':
        # out = outputs.sum_sum_sum_eres_cw
    # else:
        # out = outputs.eres
    # out.AD1.plot_hist(label='JUNO')
    # step('_04_eres')


# if args.show:
    # P.show()

# #
# # Dump the histogram to a dot graph
# #
# if args.dot:
    # try:
        # from gna.graphviz import GNADot

        # graph = GNADot(context.outputs.ee, joints=False)
        # graph.write(args.dot)
        # print( 'Write output to:', args.dot )

        # graph = GNADot(context.outputs.thermal_power.values(), joints=False)
        # name = args.dot.replace('.dot', '_lt.dot')
        # graph.write(name)
        # print( 'Write output to:', name )
    # except Exception as e:
        # print( '\033[31mFailed to plot dot\033[0m' )
        # raise

