# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R

seconds_per_day = 60*60*24
class exp(baseexp):
    detectorname = 'AD1'

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument( '-o', '--output', help='output figure name' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-e', '--embed', action='store_true', help='embed')
        parser.add_argument('-c', '--composition', default='complete', choices=['complete', 'minimal'], help='Set the indices coverage')
        parser.add_argument('-m', '--mode', default='simple', choices=['simple'], help='Set the topology')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')

    def __init__(self, namespace, opts):
        baseexp.__init__(self, namespace, opts)

        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.build()
        self.register()

        if self.opts.stats:
            self.print_stats()

    def init_nidx(self):
        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
        ]
        if self.opts.composition=='minimal':
            self.nidx[1][2] = self.nidx[1][2][:1]
            self.nidx[2][2] = self.nidx[2][2][:1]

        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        self.formula = self.formula_base
        versions = dict(
                simple = self.formula_ibd_simple,
                )
        self.formula.append(versions[self.opts.mode])
        self.formula+=self.formula_back

    def init_configuration(self):
        self.cfg = NestedDict(
                kinint2 = NestedDict(
                    bundle   = dict(name='integral_2d1d', version='v02', names=dict(integral='kinint2')),
                    variables = ('evis', 'ctheta'),
                    edges    = np.arange(0.6, 12.001, 0.01),
                    #  edges    = np.linspace(0.0, 12.001, 601),
                    xorders   = 4,
                    yorder   = 5,
                    ),
                ibd_xsec = NestedDict(
                    bundle = dict(name='xsec_ibd', version='v02'),
                    order = 1,
                    ),
                oscprob = NestedDict(
                    bundle = dict(name='oscprob', version='v02', major='rdc'),
                    ),
                anuspec = NestedDict(
                    bundle = dict(name='reactor_anu_spectra', version='v03'),
                    name = 'anuspec',
                    filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                        'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                    # strategy = dict( underflow='constant', overflow='extrapolate' ),
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.025 ), [ 12.3 ] ) ),
                    ),
                anuspec_1 = NestedDict(
                    bundle = dict(name='reactor_anu_spectra', version='v03'),
                    name = 'anuspec',
                    filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                                'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                    # strategy = dict( underflow='constant', overflow='extrapolate' ),
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.05 ), [ 12.3 ] ) ),
                    ),
                eff = NestedDict(
                    bundle = dict(
                        name='parameters',
                        version='v01'),
                    parameter="eff",
                    label='Detection efficiency',
                    pars = uncertain(0.8, 'fixed')
                    ),
                global_norm = NestedDict(
                    bundle = dict(
                        name='parameters',
                        version='v01'),
                    parameter="global_norm",
                    label='Global normalization',
                    pars = uncertain(1, 'free'),
                    ),
                fission_fractions = NestedDict(
                    bundle = dict(name="parameters",
                        version = "v01",
                        major = 'i'
                        ),
                    parameter = "fission_fractions",
                    label = 'Fission fraction of {isotope} in reactor {reactor}',
                    objectize=True,
                    pars = uncertaindict([
                        ('U235',  0.60),
                        ('Pu239', 0.27),
                        ('U238',  0.07),
                        ('Pu241', 0.06)
                        ],
                        # uncertainty = 30.0,
                        # mode = 'percent',
                        mode = 'fixed',
                        ),
                    ),
                livetime = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "livetime",
                        label = 'Livetime of {detector} in seconds',
                        pars = uncertaindict(
                            [('AD1', (6*365*seconds_per_day, 'fixed'))],
                            ),
                        ),
                baselines = NestedDict(
                        bundle = dict(name='reactor_baselines', version='v01', major = 'rd'),
                        reactors  = 'data/juno_nominal/coordinates_reactors.py',
                        detectors = 'data/juno_nominal/coordinates_det.py',
                        unit = 'km'
                        ),
                thermal_power = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "thermal_power",
                        label = 'Thermal power of {reactor} in GWt',
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
                        ),
                target_protons = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "target_protons",
                        label = 'Number of protons in {detector}',
                        pars = uncertaindict(
                            [('AD1', (1.42e33, 'fixed'))],
                            ),
                        ),
                conversion_factor =  NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter='conversion_factor',
                        label='Conversion factor from GWt to MeV',
                        #taken from transformations/neutrino/ReactorNorm.cc
                        pars = uncertain(R.NeutrinoUnits.reactorPowerConversion, 'fixed'),
                        ),
                eper_fission =  NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "eper_fission",
                        label = 'Energy per fission for {isotope} in MeV',
                        objectize = True,
                        pars = uncertaindict(
                            [
                              # ('U235',  (201.92, 0.46)),
                              # ('U238',  (205.52, 0.96)),
                              # ('Pu239', (209.99, 0.60)),
                              # ('Pu241', (213.60, 0.65))
                              ('U235',  201.92),
                              ('U238',  205.52),
                              ('Pu239', 209.99),
                              ('Pu241', 213.60)
                              ],
                            # mode='absolute'
                            mode='fixed'
                            ),
                        ),
                eres = NestedDict(
                        bundle = dict(name='detector_eres_normal', version='v01', major=''),
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                        pars = uncertaindict([
                            ('a', (0.000, 'fixed')) ,
                            ('b', (0.03, 'fixed')) ,
                            ('c', (0.000, 'fixed'))
                            ]),
                        expose_matrix = False
                        ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v03', major=''),
                        rounding = 3,
                        edges = np.concatenate(( [0.7], np.arange(1, 8.001, 0.02), [9.0, 12.0] )),
                        name = 'rebin',
                        label = 'Final histogram\n{detector}'
                        ),
                )

    def build(self):
        from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
        from gna.bundle import execute_bundles

        # Initialize the expression and indices
        self.expression = Expression_v01(self.formula, self.nidx)

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
        self.context = ExpressionContext_v01(self.cfg, ns=self.namespace)
        self.expression.build(self.context)

        if self.opts.verbose>1:
            width = 40
            print('Outputs:')
            print(self.context.outputs.__str__(nested=True, width=width))

            print()
            print('Inputs:')
            print(self.context.inputs.__str__(nested=True, width=width))

            print()

        if self.opts.verbose or self.opts.stats:
            print('Parameters:')
            self.stats = dict()
            self.namespace.printparameters(labels=True, stats=self.stats)

    def register(self):
        ns = self.namespace
        outputs = self.context.outputs
        #  ns.addobservable("{0}_unoscillated".format(self.detectorname), outputs, export=False)
        ns.addobservable("{0}_noeffects".format(self.detectorname),    outputs.kinint2.AD1, export=False)
        ns.addobservable("Enu",    outputs.enu, export=False)
        ns.addobservable("{0}_fine".format(self.detectorname),         outputs.eres.AD1)
        ns.addobservable("{0}".format(self.detectorname),              outputs.rebin.AD1)

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.rebin.AD1
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    formula_base = [
            'baseline[d,r]',
            'enu| ee(evis()), ctheta()',
            'livetime[d]',
            'eper_fiss_transform = eper_fission[i]()',
            'conversion_factor',
            'numerator = eff * livetime[d] * thermal_power[r] * '
                 'fission_fractions[r,i]() * conversion_factor * target_protons[d] ',
            'eper_fission_avg = sum[i] | eper_fiss_transform * fission_fractions[r,i]',
            'power_livetime_factor = numerator / eper_fission_avg',
            # Detector effects
            'eres_matrix| evis_hist()'
    ]

    formula_ibd_simple = '''ibd =
                            eres|
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
            'observation=rebin| ibd'
            ]

    lib = dict(
            cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                           label='anu count rate | {isotope}@{reactor}-\\>{detector} ({component})'),
            cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
            cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),

            ibd                     = dict(expr='eres', label='Observed IBD spectrum | {detector}'),
            ibd_noeffects           = dict(expr='kinint2', label='Observed IBD spectrum (no effects) | {detector}'),

            oscprob_weighted        = dict(expr='oscprob*pmns'),
            oscprob_full            = dict(expr='sum:c|oscprob_weighted',
                # label='anue survival probability | weight: {weight_label}'
                label='anue survival probability | weight: ???'
                ),
            eper_fiss_transform     = dict(expr='eper_fission_transform',
                                           label='eper_fission for {isotope}' ),

            fission_fractions       = dict(expr='fission_fractions[r,i]()',
                                           label="Fission fraction for {isotope} in reactor {reactor}"),
            eper_fission_weight = dict(expr='eper_fission_weight',
                                           label="Weighted eper_fission for {isotope} in reactor {reactor}"),

            numerator = dict(expr='numerator', label='thermal_weight.{isotope}.{reactor}'),
            eper_fission_avg = dict(expr='eper_fission_avg', label='Sum over all isotopes weighted eper_fission  | for {reactor}'),
            power_lifetime_factor =   dict(expr='power_lifetime_factor'),
            anuspec_weighted        = dict(expr='anuspec*power_livetime_factor'),
            anuspec_rd              = dict(expr='sum:i|anuspec_weighted',
                # label='anue spectrum {reactor}-\\>{detector} | weight: {weight_label}'
                label='anue spectrum {reactor}-\\>{detector} | weight: ???'
                ),

            countrate_rd            = dict(expr='anuspec_rd*ibd_xsec*jacobian*oscprob_full'),
            countrate_weighted      = dict(expr='baselineweight*countrate_rd'),
            countrate               = dict(expr='sum:r|countrate_weighted', label='Count rate {detector} | weight: {weight_label}'),

            observation_raw         = dict(expr='bkg+ibd', label='Observed spectrum | {detector}'),

            iso_spectrum_w          = dict(expr='kinint2*power_livetime_factor'),
            reac_spectrum           = dict(expr='sum:i|iso_spectrum_w'),
            reac_spectrum_w         = dict(expr='baselineweight*reac_spectrum'),
            ad_spectrum_c           = dict(expr='sum:r|reac_spectrum_w'),
            ad_spectrum_cw          = dict(expr='pmns*ad_spectrum_c'),
            ad_spectrum_w           = dict(expr='sum:c|ad_spectrum_cw'),

            eres_cw           = dict(expr='eres*pmns'),
            )

