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
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')
        parser.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres', 'multieres'], default=['lsnl', 'eres'], help='Energy model components')
        correlations = [ 'lsnl', 'subdetectors' ]
        parser.add_argument('--correlation',  nargs='*', default=correlations, choices=correlations, help='Enable correalations')

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
        self.subdetectors_number = 200
        self.subdetectors_names = ['subdet%03i'%i for i in range(self.subdetectors_number)]
        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('s', 'subdetector', self.subdetectors_names)
        ]
        if self.opts.composition=='minimal':
            self.nidx[1][2] = self.nidx[1][2][:1]
            self.nidx[2][2] = self.nidx[2][2][:1]

        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        if 'eres' in self.opts.energy_model and 'multieres' in self.opts.energy_model:
            raise Exception('Energy model options "eres" and "multieres" are mutually exclusive: use only one of them')

        self.formula = list(self.formula_base)

        energy_model_formula = ''
        energy_model = self.opts.energy_model
        if 'lsnl' in energy_model:
            energy_model_formula = 'lsnl| '
            self.formula.append('evis_edges_hist| evis_hist()')
        if 'eres' in energy_model:
            energy_model_formula = 'eres| '+energy_model_formula
            self.formula.append('eres_matrix| evis_hist()')
        elif 'multieres' in energy_model:
            energy_model_formula = 'sum[s]| subdetector_fraction[s] * eres[s]| '+energy_model_formula
            self.formula.append('eres_matrix[s]| evis_hist()')

        self.formula.append('ibd=' + energy_model_formula + self.formula_ibd_noeffects)
        self.formula+=self.formula_back

    def init_configuration(self):
        self.cfg = NestedDict(
                kinint2 = NestedDict(
                    bundle   = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2')),
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
                lsnl = NestedDict(
                        bundle = dict( name='energy_nonlinearity_birks_cherenkov', version='v01', major=''),
                        stopping_power='stoppingpower.txt',
                        annihilation_electrons=dict(
                            file='input/hgamma2e.root',
                            histogram='hgamma2e_1KeV',
                            scale=1.0/50000 # events simulated
                            ),
                        pars = uncertaindict(
                            [
                                ('birks.Kb0',               (1.0, 'fixed')),
                                ('birks.Kb1',           (15.2e-3, 0.1776)),
                                # ('birks.Kb2',           (0.0, 'fixed')),
                                ("cherenkov.E_0",         (0.165, 'fixed')),
                                ("cherenkov.p0",  ( -7.26624e+00, 'fixed')),
                                ("cherenkov.p1",   ( 1.72463e+01, 'fixed')),
                                ("cherenkov.p2",  ( -2.18044e+01, 'fixed')),
                                ("cherenkov.p3",   ( 1.44731e+01, 'fixed')),
                                ("cherenkov.p4",   ( 3.22121e-02, 'fixed')),
                                ("Npescint",            (1341.38, 0.0059)),
                                ("kC",                      (0.5, 0.4737)),
                                ("normalizationEnergy",   (12.0, 'fixed'))
                                ],
                            mode='relative'
                            ),
                        integration_order = 2,
                        correlations_pars = [ 'birks.Kb1', 'Npescint', 'kC' ],
                        correlations = [ 1.0,   0.94, -0.97,
                                         0.94,  1.0,  -0.985,
                                        -0.97, -0.985, 1.0   ],
                        fill_matrix=True,
                        labels = dict(
                            normalizationEnergy = 'Conservative normalization point at 12 MeV'
                            ),
                        ),
                eres = NestedDict(
                        bundle = dict(name='detector_eres_normal', version='v01', major='', inactive='multieres' in self.opts.energy_model),
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                        pars = uncertaindict([
                            ('a', (0.000, 'fixed')) ,
                            ('b', (0.03, 'fixed')) ,
                            ('c', (0.000, 'fixed'))
                            ]),
                        expose_matrix = False
                        ),
                subdetector_fraction = NestedDict(
                        bundle = dict(name="parameters", version = "v02"),
                        parameter = "subdetector_fraction",
                        label = 'Subdetector fraction weight for {subdetector}',
                        pars = uncertaindict(
                            [(subdet_name, (1.0/self.subdetectors_number, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
                            ),
                        correlations = 'covariance/corrmap_xuyu.txt',
                        verbose=2
                        ),
                multieres = NestedDict(
                        bundle = dict(name='detector_multieres_stats', version='v01', major='s', inactive='multieres' not in self.opts.energy_model),
                        # pars: sigma_e/e = sqrt(b^2/E),
                        parameter = 'eres',
                        nph = 'subdetector200_nph.txt',
                        expose_matrix = False
                        ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v03', major=''),
                        rounding = 3,
                        edges = np.concatenate(( [0.7], np.arange(1, 8.001, 0.02), [9.0, 12.0] )),
                        name = 'rebin',
                        label = 'Final histogram {detector}'
                        ),
                )
        if not 'lsnl' in self.opts.correlation:
            self.cfg.lsnl.correlations = None
            self.cfg.lsnl.correlations_pars = None

        if not 'subdetectors' in self.opts.correlation:
            self.cfg.subdetector_fraction.correlations = None

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

        if self.opts.verbose>2:
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
        ns.addobservable("Enu",    outputs.enu, export=False)
        ns.addobservable("{0}_noeffects".format(self.detectorname),    outputs.kinint2.AD1)
        fine = outputs.kinint2.AD1

        if 'lsnl' in self.opts.energy_model:
            ns.addobservable("{0}_lsnl".format(self.detectorname),     outputs.lsnl.AD1)
            fine = outputs.lsnl.AD1

        if 'eres' in self.opts.energy_model:
            ns.addobservable("{0}_eres".format(self.detectorname),     outputs.eres.AD1)
            fine = outputs.eres.AD1

        if 'multieres' in self.opts.energy_model:
            ns.addobservable("{0}_eres".format(self.detectorname),     outputs.ibd.AD1)
            fine = outputs.ibd.AD1

        ns.addobservable("{0}_fine".format(self.detectorname),         fine)
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
            'conversion_factor',
            'numerator = eff * livetime[d] * thermal_power[r] * '
                 'fission_fractions[r,i]() * conversion_factor * target_protons[d] ',
            'eper_fission_avg = sum[i] | eper_fission[i] * fission_fractions[r,i]()',
            'power_livetime_factor = numerator / eper_fission_avg',
    ]

    formula_ibd_noeffects = '''
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

            eres_weighted           = dict(expr='subdetector_fraction*eres', label='{{Fractional observed spectrum {subdetector}|weight: {weight_label}}}'),
            ibd                     = dict(expr=('eres', 'sum:c|eres_weighted'), label='Observed IBD spectrum | {detector}'),
            ibd_noeffects           = dict(expr='kinint2', label='Observed IBD spectrum (no effects) | {detector}'),

            oscprob_weighted        = dict(expr='oscprob*pmns'),
            oscprob_full            = dict(expr='sum:c|oscprob_weighted', label='anue survival probability | weight: {weight_label}'),

            fission_fractions       = dict(expr='fission_fractions[r,i]()', label="Fission fraction for {isotope} at {reactor}"),
            eper_fission_weight     = dict(expr='eper_fission_weight', label="Weighted eper_fission for {isotope} at {reactor}"),
            eper_fission_weighted   = dict(expr='eper_fission*fission_fractions', label="{{Energy per fission for {isotope} | weighted with fission fraction at {reactor}}}"),

            eper_fission_avg        = dict(expr='eper_fission_avg', label='Average energy per fission at {reactor}'),
            power_livetime_factor   = dict(expr='power_livetime_factor', label='{{Power-livetime factor (~nu/s)|{reactor}.{isotope}-\\>{detector}}}'),
            numerator               = dict(expr='numerator', label='{{Power-livetime factor (~MW)|{reactor}.{isotope}-\\>{detector}}}'),
            power_livetime_scale    = dict(expr='eff*livetime*thermal_power*conversion_factor*target_protons', label='{{Power-livetime factor (~MW)| {reactor}.{isotope}-\\>{detector}}}'),
            anuspec_weighted        = dict(expr='anuspec*power_livetime_factor', label='{{Antineutrino spectrum|{reactor}.{isotope}-\\>{detector}}}'),
            anuspec_rd              = dict(expr='sum:i|anuspec_weighted', label='{{Antineutrino spectrum|{reactor}-\\>{detector}}}'),

            countrate_rd            = dict(expr='anuspec_rd*ibd_xsec*jacobian*oscprob_full', label='Countrate {reactor}-\\>{detector}'),
            countrate_weighted      = dict(expr='baselineweight*countrate_rd'),
            countrate               = dict(expr='sum:r|countrate_weighted', label='{{Count rate at {detector}|weight: {weight_label}}}'),

            observation_raw         = dict(expr='bkg+ibd', label='Observed spectrum | {detector}'),

            iso_spectrum_w          = dict(expr='kinint2*power_livetime_factor'),
            reac_spectrum           = dict(expr='sum:i|iso_spectrum_w'),
            reac_spectrum_w         = dict(expr='baselineweight*reac_spectrum'),
            ad_spectrum_c           = dict(expr='sum:r|reac_spectrum_w'),
            ad_spectrum_cw          = dict(expr='pmns*ad_spectrum_c'),
            ad_spectrum_w           = dict(expr='sum:c|ad_spectrum_cw'),

            eres_cw           = dict(expr='eres*pmns'),
            )

