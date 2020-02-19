# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna import constructors as C
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R

seconds_per_day = 60*60*24
class exp(baseexp):
    """
JUNO experiment implementation

Derived [2019.12] from:
    - Daya Bay model from dybOscar and GNA
    - juno_chengyp

Changes since previous implementation [juno_chengyp]:
    - Dropped Enu-mode support
    - WIP: add matter oscillations

Implements:
    - Reactor antineutrino flux:
      * Spectra:
        + Huber+Mueller
        + ILL+Vogel
      * [optional] Off-equilibrium corrections (Mueller)
      * NO SNF contribution
    - Vacuum 3nu oscillations
    - Evis mode with 2d integration (similary to dybOscar)
    - [optional] Birks-Cherenkov detector energy responce (Yaping)
    - [optional] Detector energy resolution
    - [optional] Multi-detector energy resolution (Yaping)
        * subdetectors summed togather
        * subdetectors concatenated

Misc changes:
    - Switch oscillation probability bundle from v03 to v04 (OscProb3 class)
    - Switch to double angle parameters for theta12 and theta13
    - Added concatenated subdetectors
    - Uncomment uncertainties:
      * energy per fission
      * fission fractions
    """

    detectorname = 'AD1'

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument( '-o', '--output', help='output figure name' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')

        # Energy model
        parser.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres', 'multieres'], default=['lsnl', 'eres'], help='Energy model components')
        parser.add_argument('--subdetectors-number', type=int, choices=(200, 5), help='Number of subdetectors (multieres mode)')
        parser.add_argument('--multieres', default='sum', choices=['sum', 'concat'], help='How to treat subdetectors (multieres mode)')
        parser.add_argument('--eres-b-relsigma', type=float, help='Energy resolution parameter (b) relative uncertainty')

        eres = parser.add_mutually_exclusive_group()
        eres.add_argument('--eres-sigma', type=float, help='Energy resolution at 1 MeV')
        eres.add_argument('--eres-npe', type=float, default=1200.0, help='Average Npe at 1 MeV')

        # binning
        parser.add_argument('--estep', default=0.02, choices=[0.02, 0.01], type=float, help='Binning step')

        # reactor flux
        parser.add_argument('--reactors', choices=['single', 'near-equal', 'far-off', 'pessimistic', 'nohz', 'dayabay'], default=[], nargs='+', help='reactors options')
        parser.add_argument('--flux', choices=['huber-mueller', 'ill-vogel'], default='huber-mueller', help='Antineutrino flux')
        parser.add_argument('--offequilibrium-corr', action='store_true', help="Turn on offequilibrium correction to antineutrino spectra")

        # osc prob
        parser.add_argument('--oscprob', choices=['vacuum', 'matter'], default='vacuum', help='oscillation probability type')

        # Parameters
        parser.add_argument('--free', choices=['minimal', 'osc'], default='minimal', help='free oscillation parameterse')
        parser.add_argument('--parameters', choices=['default', 'yb', 'yb_t12', 'yb_t12_t13', 'yb_t12_t13_dm12', 'global'], default='default', help='set of parameters to load')
        parser.add_argument('--dm', default='ee', choices=('23', 'ee'), help='Δm² parameter to use')
        parser.add_argument('--pdgyear', choices=[2016, 2018], default=2018, type=int, help='PDG version to read the oscillation parameters')
        parser.add_argument('--spectrum-unc', choices=['initial', 'final', 'none'], default='none', help='type of the spectral uncertainty')
        correlations = [ 'lsnl', 'subdetectors' ]
        parser.add_argument('--correlation',  nargs='*', default=correlations, choices=correlations, help='Enable correalations')

    def init(self):
        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.preinit_variables()
        self.build()
        self.parameters()
        self.register()
        self.autodump()

        if self.opts.stats:
            self.print_stats()

    def init_nidx(self):
        if self.opts.subdetectors_number:
            self.subdetectors_names = ['subdet%03i'%i for i in range(self.opts.subdetectors_number)]
        else:
            self.subdetectors_names = ()
        self.reactors = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']
        if 'pessimistic' in self.opts.reactors:
            self.reactors.remove('TS3')
            self.reactors.remove('TS4')
        if 'far-off' in self.opts.reactors:
            self.reactors.remove('DYB')
            self.reactors.remove('HZ')
        if 'nohz' in self.opts.reactors:
            self.reactors.remove('HZ')
        if 'dayabay' in self.opts.reactors:
            self.reactors=['DYB']
        if 'single' in self.opts.reactors:
            self.reactors=['YJ1']

        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     self.reactors],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('s', 'subdetector', self.subdetectors_names)
        ]
        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        if 'eres' in self.opts.energy_model and 'multieres' in self.opts.energy_model:
            raise Exception('Energy model options "eres" and "multieres" are mutually exclusive: use only one of them')

        self.formula = list(self.formula_base)

        oscprob_part = self.opts.oscprob=='vacuum' and self.formula_oscprob_vacuum or self.formula_oscprob_matter
        offeq_correction = '*offeq_correction[i,r](enu())' if self.opts.offequilibrium_corr else ''
        ibd = self.formula_ibd_noeffects.format(oscprob=oscprob_part, offeq_correction=offeq_correction)
        self.formula = self.formula + self.formula_enu

        energy_model_formula = ''
        energy_model = self.opts.energy_model
        concat_subdetectors=False
        if 'lsnl' in energy_model:
            energy_model_formula = 'lsnl| '
            self.formula.append('evis_edges_hist| evis_hist')
        if 'eres' in energy_model:
            energy_model_formula = 'eres| '+energy_model_formula
            self.formula.append('eres_matrix| evis_hist')
        elif 'multieres' in energy_model:
            if self.opts.multieres=='sum':
                energy_model_formula = 'sum[s]| subdetector_fraction[s] * eres[s]| '+energy_model_formula
                self.formula.append('eres_matrix[s]| evis_hist')
            elif self.opts.multieres=='concat':
                energy_model_formula = 'concat[s]| rebin| subdetector_fraction[s] * eres[s]| '+energy_model_formula
                self.formula.append('eres_matrix[s]| evis_hist')
                concat_subdetectors = True

        if concat_subdetectors:
            formula_back = 'observation=norm * ibd'
        else:
            formula_back = 'observation=norm * rebin(ibd)'

        if self.opts.spectrum_unc=='initial':
            ibd = ibd+'*shape_norm()'
        elif self.opts.spectrum_unc=='final':
            formula_back = formula_back+'*shape_norm()'

        self.formula.append('ibd=' + energy_model_formula + ibd)
        self.formula.append(formula_back)

    def parameters(self):
        ns = self.namespace
        dmxx = 'pmns.DeltaMSq'+str(self.opts.dm).upper()
        if self.opts.free=='minimal':
            fixed_pars = ['pmns.SinSqDouble13', 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']
            free_pars  = [dmxx]
        elif self.opts.free=='osc':
            fixed_pars = []
            free_pars  = [dmxx, 'pmns.SinSqDouble13', 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']
        else:
            raise Exception('Unsupported option')

        for par in fixed_pars:
            ns[par].setFixed()
        for par in free_pars:
            ns[par].setFree()

        def single2double(v):
            return 4.0*v*(1.0-v)
        if self.opts.parameters=='yb':
            ns['pmns.SinSqDouble12'].setCentral(single2double(0.307))
            ns['pmns.SinSqDouble13'].setCentral(0.094)
            ns['pmns.DeltaMSq12'].setCentral(7.54e-5)
            ns['pmns.DeltaMSqEE'].setCentral(2.43e-3)
            ns['pmns.SinSqDouble12'].reset()
            ns['pmns.SinSqDouble13'].reset()
            ns['pmns.DeltaMSq12'].reset()
            ns['pmns.DeltaMSqEE'].reset()
        elif self.opts.parameters=='yb_t12':
            ns['pmns.SinSqDouble13'].setCentral(0.094)
            ns['pmns.DeltaMSq12'].setCentral(7.54e-5)
            ns['pmns.DeltaMSqEE'].setCentral(2.43e-3)
            ns['pmns.SinSqDouble13'].reset()
            ns['pmns.DeltaMSq12'].reset()
            ns['pmns.DeltaMSqEE'].reset()
        elif self.opts.parameters=='yb_t12_t13':
            ns['pmns.DeltaMSq12'].setCentral(7.54e-5)
            ns['pmns.DeltaMSqEE'].setCentral(2.43e-3)
            ns['pmns.DeltaMSq12'].reset()
            ns['pmns.DeltaMSqEE'].reset()
        elif self.opts.parameters=='yb_t12_t13_dm12':
            ns['pmns.DeltaMSqEE'].setCentral(2.43e-3)
            ns['pmns.DeltaMSqEE'].reset()
        elif self.opts.parameters=='global':
            ns['pmns.DeltaMSq12'].setCentral(7.39e-5)
            ns['pmns.DeltaMSq12'].reset()

    def init_configuration(self):
        if self.opts.eres_npe:
            self.opts.eres_sigma = self.opts.eres_npe**-0.5
        else:
            self.opts.eres_npe = self.opts.eres_sigma**-2
        print('Energy resolution at 1 MeV: {}% ({} pe)'.format(self.opts.eres_sigma*100, self.opts.eres_npe))

        self.cfg = NestedDict(
                kinint2 = NestedDict(
                    bundle   = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2')),
                    variables = ('evis', 'ctheta'),
                    edges    = np.arange(0.6, 12.001, 0.01),
                    #  edges    = np.linspace(0.0, 12.001, 601),
                    xorders   = 4,
                    yorder   = 5,
                    ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v03', major=''),
                        rounding = 3,
                        edges = np.concatenate( (
                                    [0.7],
                                    np.arange(1, 6.0, self.opts.estep),
                                    np.arange(6, 7.0, 0.1),
                                    [7.0, 7.5, 12.0]
                                )
                            ),
                        name = 'rebin',
                        label = 'Final histogram {detector}'
                        ),
                ibd_xsec = NestedDict(
                    bundle = dict(name='xsec_ibd', version='v02'),
                    order = 1,
                    ),
                oscprob = NestedDict(
                    bundle = dict(name='oscprob', version='v04', major='rdc', inactive=self.opts.oscprob=='matter'),
                    pdgyear = self.opts.pdgyear,
                    dm      = self.opts.dm
                    ),
                oscprob_matter = NestedDict(
                    bundle = dict(name='oscprob_matter', version='v01', major='rd', inactive=self.opts.oscprob=='vacuum',
                                  names=dict(oscprob='oscprob_matter')),
                    density = 2.6, # g/cm3
                    pdgyear = self.opts.pdgyear,
                    dm      = self.opts.dm
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
                offeq_correction = NestedDict(
                    bundle = dict(name='reactor_offeq_spectra',
                                  version='v03', major='ir'),
                    offeq_data = 'data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
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
                    bundle = dict(name="parameters_yaml_v01", major = 'i'),
                    parameter = "fission_fractions",
                    label = 'Fission fraction of {isotope} in reactor {reactor}',
                    objectize=True,
                    data = 'data/dayabay/reactor/fission_fraction/2013.12.05_xubo.yaml'
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
                        reactors  = 'near-equal' in self.opts.reactors \
                                     and 'data/juno_nominal/coordinates_reactors_equal.py' \
                                     or 'data/juno_nominal/coordinates_reactors.py',
                        detectors = 'data/juno_nominal/coordinates_det.py',
                        unit = 'km'
                        ),
                norm = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "norm",
                        label = 'Reactor power/detection efficiency correlated normalization',
                        pars = uncertain(1.0, (2**2+1**2)**0.5, 'percent')
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
                            uncertainty=0.8,
                            mode='percent'
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
                              ('U235',  (201.92, 0.46)),
                              ('U238',  (205.52, 0.96)),
                              ('Pu239', (209.99, 0.60)),
                              ('Pu241', (213.60, 0.65))
                              ],
                            mode='absolute'
                            ),
                        ),
                lsnl = NestedDict(
                        bundle = dict( name='energy_nonlinearity_birks_cherenkov', version='v01', major=''),
                        stopping_power='data/data_juno/energy_model/2019_birks_cherenkov_v01/stoppingpower.txt',
                        annihilation_electrons=dict(
                            file='data/data_juno/energy_model/2019_birks_cherenkov_v01/hgamma2e.root',
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
                shape_uncertainty = NestedDict(
                        unc = uncertain(1.0, 1.0, 'percent'),
                        nbins = 200 # number of bins, the uncertainty is defined to
                        )
                )

        if 'eres' in self.opts.energy_model:
            bconf = self.opts.eres_b_relsigma and (self.opts.eres_b_relsigma, 'relative') or ('fixed',)
            self.cfg.eres = NestedDict(
                    bundle = dict(name='detector_eres_normal', version='v01', major=''),
                    # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                    parameter = 'eres',
                    pars = uncertaindict([
                        ('a', (0.000, 'fixed')) ,
                        ('b', (self.opts.eres_sigma,)+bconf) ,
                        ('c', (0.000, 'fixed'))
                        ]),
                    expose_matrix = False
                    )
        elif 'multieres' in self.opts.energy_model:
            if self.opts.subdetectors_number==200:
                self.cfg.subdetector_fraction = NestedDict(
                        bundle = dict(name="parameters", version = "v02"),
                        parameter = "subdetector_fraction",
                        label = 'Subdetector fraction weight for {subdetector}',
                        pars = uncertaindict(
                            [(subdet_name, (1.0/self.opts.subdetectors_number, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
                            ),
                        correlations = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/corrmap_xuyu.txt'
                        )
                self.cfg.multieres = NestedDict(
                        bundle = dict(name='detector_multieres_stats', version='v01', major='s'),
                        # pars: sigma_e/e = sqrt(b^2/E),
                        parameter = 'eres',
                        relsigma = self.opts.eres_b_relsigma,
                        nph = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector200_nph.txt',
                        rescale_nph = self.opts.eres_npe,
                        expose_matrix = False
                        )
            elif self.opts.subdetectors_number==5:
                self.cfg.subdetector_fraction = NestedDict(
                        bundle = dict(name="parameters", version = "v03"),
                        parameter = "subdetector_fraction",
                        label = 'Subdetector fraction weight for {subdetector}',
                        pars = uncertaindict(
                            [(subdet_name, (1.0/self.opts.subdetectors_number, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
                            ),
                        covariance = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_cov.txt'
                        )
                self.cfg.multieres = NestedDict(
                        bundle = dict(name='detector_multieres_stats', version='v01', major='s'),
                        # pars: sigma_e/e = sqrt(b^2/E),
                        parameter = 'eres',
                        relsigma = self.opts.eres_b_relsigma,
                        nph = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_nph.txt',
                        rescale_nph = self.opts.eres_npe,
                        expose_matrix = False
                        )
            else:
                assert False

        if not 'lsnl' in self.opts.correlation:
            self.cfg.lsnl.correlations = None
            self.cfg.lsnl.correlations_pars = None
        if not 'subdetectors' in self.opts.correlation:
            self.cfg.subdetector_fraction.correlations = None

        self.cfg.eff.pars = uncertain(0.73, 'fixed')
        self.cfg.livetime.pars['AD1'] = uncertain( 6*330*seconds_per_day, 'fixed' )

    def preinit_variables(self):
        if self.opts.spectrum_unc in ['final', 'initial']:
            spec = self.namespace('spectrum')
            cfg = self.cfg.shape_uncertainty
            unc = cfg.unc

            if self.opts.spectrum_unc=='initial':
                edges = self.cfg.kinint2.edges
            elif self.opts.spectrum_unc=='final':
                edges = self.cfg.rebin.edges

            # bin-to-bin should take into account the number of bins it is applied to
            unccorrection = ((edges.size-1.0)/cfg.nbins)**0.5
            unc.uncertainty*=unccorrection

            names = []
            for bini in range(edges.size-1):
                name = 'norm_bin_%04i'%bini
                names.append(name)
                label = 'Spectrum shape unc. final bin %i (%.03f, %.03f) MeV'%(bini, edges[bini], edges[bini+1])
                spec.reqparameter(name, cfg=unc, label=label)

            with spec:
                vararray = C.VarArray(names, labels='Spectrum shape norm')

            self.cfg.shape_uncertainty = NestedDict(
                    bundle = dict(name='predefined', version='v01'),
                    name   = 'shape_norm',
                    inputs = None,
                    outputs = vararray.single(),
                    unc    = cfg.unc,
                    object = vararray
                    )
        elif self.opts.spectrum_unc=='none':
            pass
        else:
            raise Exception('Unknown spectrum shape uncertainty type: '+self.opts.spectrum_unc)

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

    def autodump(self):
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
            correlations = self.opts.verbose>2 and 'full' or 'short'
            self.namespace.printparameters(labels=True, stats=self.stats, correlations=correlations)

    def register(self):
        ns = self.namespace
        outputs = self.context.outputs
        #  ns.addobservable("{0}_unoscillated".format(self.detectorname), outputs, export=False)
        ns.addobservable("Enu",    outputs.enu, export=False)

        if 'ibd_noeffects_bf' in outputs:
            ns.addobservable("{0}_noeffects".format(self.detectorname),    outputs.ibd_noeffects_bf.AD1)
            fine = outputs.ibd_noeffects_bf.AD1
        else:
            ns.addobservable("{0}_noeffects".format(self.detectorname),    outputs.kinint2.AD1)
            fine = outputs.kinint2.AD1

        if 'lsnl' in self.opts.energy_model:
            ns.addobservable("{0}_lsnl".format(self.detectorname),     outputs.lsnl.AD1)
            fine = outputs.lsnl.AD1

        if 'eres' in self.opts.energy_model:
            ns.addobservable("{0}_eres".format(self.detectorname),     outputs.eres.AD1)
            fine = outputs.eres.AD1

        if 'multieres' in self.opts.energy_model:
            if self.opts.multieres=='concat' and self.opts.subdetectors_number<10:
                sns = ns('{}_sub'.format(self.detectorname))
                for i, out in enumerate(outputs.rebin.AD1.values()):
                    sns.addobservable("sub{:02d}".format(i), out)
            ns.addobservable("{0}_eres".format(self.detectorname),     outputs.ibd.AD1)
            fine = outputs.ibd.AD1

        ns.addobservable("{0}_fine".format(self.detectorname),         fine)
        ns.addobservable("{0}".format(self.detectorname),              outputs.observation.AD1)

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.rebin.AD1
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    formula_enu = ['evis_hist=evis_hist()', 'enu| ee(evis()), ctheta()']

    formula_base = [
            'baseline[d,r]',
            'livetime[d]',
            'conversion_factor',
            'numerator = eff * livetime[d] * thermal_power[r] * '
                 'fission_fractions[r,i]() * conversion_factor * target_protons[d] ',
            'eper_fission_avg = sum[i] | eper_fission[i] * fission_fractions[r,i]()',
            'power_livetime_factor = numerator / eper_fission_avg',
    ]

    formula_ibd_noeffects = '''
                            kinint2(
                              sum[r]|
                                baselineweight[r,d]*
                                ibd_xsec(enu(), ctheta())*
                                jacobian(enu(), ee(), ctheta())*
                                expand(sum[i]|
                                power_livetime_factor*anuspec[i](enu()){offeq_correction})*
                                {oscprob}
                            )
            '''

    formula_oscprob_vacuum = 'sum[c]| pmns[c]*oscprob[c,d,r](enu())'
    formula_oscprob_matter = 'oscprob_matter[d,r](enu())'

    lib = """
        cspec_diff:
          expr: 'anuspec*ibd_xsec*jacobian*oscprob'
          label: 'anu count rate | {isotope}@{reactor}-\\>{detector} ({component})'
        cspec_diff_reac_l:
          expr: 'baselineweight*cspec_diff_reac'
        cspec_diff_det_weighted:
          expr: 'pmns*cspec_diff_det'
        eres_weighted:
          expr: 'subdetector_fraction*eres'
          label: '{{Fractional observed spectrum {subdetector}|weight: {weight_label}}}'
        ibd:
          expr:
          - 'eres'
          - 'sum:c|eres_weighted'
          label: 'Observed IBD spectrum | {detector}'
        ibd_noeffects:
          expr: 'kinint2'
          label: 'Observed IBD spectrum (no effects) | {detector}'
        ibd_noeffects_bf:
          expr: 'kinint2*shape_norm'
          label: 'Observed IBD spectrum (best fit, no effects) | {detector}'
        oscprob_weighted:
          expr: 'oscprob*pmns'
        oscprob_full:
          expr: 'sum:c|oscprob_weighted'
          label: 'anue survival probability | weight: {weight_label}'
        fission_fractions:
          expr: 'fission_fractions[r,i]()'
          label: "Fission fraction for {isotope} at {reactor}"
        eper_fission_weight:
          expr: 'eper_fission_weight'
          label: "Weighted eper_fission for {isotope} at {reactor}"
        eper_fission_weighted:
          expr: 'eper_fission*fission_fractions'
          label: "{{Energy per fission for {isotope} | weighted with fission fraction at {reactor}}}"
        eper_fission_avg:
          expr: 'eper_fission_avg'
          label: 'Average energy per fission at {reactor}'
        power_livetime_factor:
          expr: 'power_livetime_factor'
          label: '{{Power-livetime factor (~nu/s)|{reactor}.{isotope}-\\>{detector}}}'
        numerator:
          expr: 'numerator'
          label: '{{Power-livetime factor (~MW)|{reactor}.{isotope}-\\>{detector}}}'
        power_livetime_scale:
          expr: 'eff*livetime*thermal_power*conversion_factor*target_protons'
          label: '{{Power-livetime factor (~MW)| {reactor}.{isotope}-\\>{detector}}}'
        anuspec_weighted:
          expr: 'anuspec*power_livetime_factor'
          label: '{{Antineutrino spectrum|{reactor}.{isotope}-\\>{detector}}}'
        anuspec_rd:
          expr: 'sum:i|anuspec_weighted'
          label: '{{Antineutrino spectrum|{reactor}-\\>{detector}}}'
        countrate_rd:
          expr:
          - 'anuspec_rd*ibd_xsec*jacobian*oscprob_full'
          - 'anuspec_rd*ibd_xsec*oscprob_full'
          label: 'Countrate {reactor}-\\>{detector}'
        countrate_weighted:
          expr: 'baselineweight*countrate_rd'
        countrate:
          expr: 'sum:r|countrate_weighted'
          label: '{{Count rate at {detector}|weight: {weight_label}}}'
        observation_raw:
          expr: 'bkg+ibd'
          label: 'Observed spectrum | {detector}'
        iso_spectrum_w:
          expr: 'kinint2*power_livetime_factor'
        reac_spectrum:
          expr: 'sum:i|iso_spectrum_w'
        reac_spectrum_w:
          expr: 'baselineweight*reac_spectrum'
        ad_spectrum_c:
          expr: 'sum:r|reac_spectrum_w'
        ad_spectrum_cw:
          expr: 'pmns*ad_spectrum_c'
        ad_spectrum_w:
          expr: 'sum:c|ad_spectrum_cw'
        eres_cw:
          expr: 'eres*pmns'
    """
