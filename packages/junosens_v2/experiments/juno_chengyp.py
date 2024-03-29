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

Derived from: Daya Bay model from dybOscar and GNA
Implements:
    - Reactor antineutrino flux: Huber+Mueller
    - NO off-equilibrium and NO SNF contribution
    - Vacuum 3nu oscillations
    - Two modes for kinematics:
        * Enu mode with 1d integration (similar to YB fitter)
        * Evis mode with 2d integration (similary to dybOscar)
    - [optional] Birks-Cherenkov detector energy responce (Yaping)
    - [optional] Detector energy resolution
    - [optional] Multi-detector energy resolution (Yaping)

The model is succeeded by juno_sensitivity_v01 model.
    """
    detectorname = 'AD1'

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument( '-o', '--output', help='output figure name' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-e', '--embed', action='store_true', help='embed')
        parser.add_argument('-c', '--composition', default='complete', choices=['complete', 'minimal'], help='Set the indices coverage')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')
        parser.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres', 'multieres'], default=['lsnl', 'eres'], help='Energy model components')
        parser.add_argument('--free', choices=['minimal', 'osc'], default='minimal', help='free oscillation parameterse')
        parser.add_argument('--mode', choices=['main', 'yb'], default='main', help='analysis mode')
        parser.add_argument('--parameters', choices=['default', 'yb', 'yb-noosc', 'junosens1'], default='default', help='set of parameters to load')
        parser.add_argument('--reactors', choices=['single', 'near-equal', 'far-off', 'pessimistic'], default=[], nargs='+', help='reactors options')
        parser.add_argument('--pdgyear', choices=[2016, 2018], default=None, type=int, help='PDG version to read the oscillation parameters')
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
        self.subdetectors_number = 200
        self.subdetectors_names = ['subdet%03i'%i for i in range(self.subdetectors_number)]
        self.reactors = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']
        if 'pessimistic' in self.opts.reactors:
            self.reactors.remove('TS3')
            self.reactors.remove('TS4')
        if 'far-off' in self.opts.reactors:
            self.reactors.remove('DYB')
            self.reactors.remove('HZ')
        if 'single' in self.opts.reactors:
            self.reactors = ['YJ1']
        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     self.reactors],
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

        mode_yb = False
        if self.opts.mode=='main':
            enu = self.formula_enu
            ibd = self.formula_ibd_noeffects
        elif self.opts.mode=='yb':
            enu = self.formula_enu_yb
            ibd = self.formula_ibd_noeffects_yb
            mode_yb = True
        else:
            raise Exception('unsupported option')
        self.formula = self.formula + enu

        energy_model_formula = ''
        energy_model = self.opts.energy_model
        if 'lsnl' in energy_model:
            energy_model_formula = 'lsnl| '
            self.formula.append('evis_edges_hist| evis_hist')
        if 'eres' in energy_model:
            energy_model_formula = 'eres| '+energy_model_formula
            self.formula.append('eres_matrix| evis_hist')
        elif 'multieres' in energy_model:
            energy_model_formula = 'sum[s]| subdetector_fraction[s] * eres[s]| '+energy_model_formula
            self.formula.append('eres_matrix[s]| evis_hist')

        formula_back = self.formula_back

        if self.opts.spectrum_unc=='initial':
            ibd = ibd+'*shape_norm()'
        elif self.opts.spectrum_unc=='final':
            formula_back = formula_back+'*shape_norm()'

        self.formula.append('ibd=' + energy_model_formula + ibd)
        self.formula.append(formula_back)

    def parameters(self):
        ns = self.namespace
        if self.opts.free=='minimal':
            fixed_pars = ['pmns.SinSq13', 'pmns.SinSq12', 'pmns.DeltaMSq12']
            free_pars  = ['pmns.DeltaMSqEE']
        elif self.opts.free=='osc':
            fixed_pars = []
            free_pars  = ['pmns.DeltaMSqEE', 'pmns.SinSq13', 'pmns.SinSq12', 'pmns.DeltaMSq12']
        else:
            raise Exception('Unsupported option')
        for par in fixed_pars:
            ns[par].setFixed()
        for par in free_pars:
            ns[par].setFree()

        if self.opts.parameters=='yb':
            ns['pmns.SinSq12'].set(0.307)
            ns['pmns.SinSq13'].set(0.024)
            ns['pmns.DeltaMSq12'].set(7.54e-5)

        if self.opts.parameters=='junosens1':
            ns['pmns.DeltaMSqEE'].set(0.0024924088)

    def init_configuration(self):
        mode_yb = self.opts.mode=='yb'
        self.cfg = NestedDict(
                kinint2 = NestedDict(
                    bundle   = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2'), inactive=mode_yb),
                    variables = ('evis', 'ctheta'),
                    edges    = np.arange(0.6, 12.001, 0.01),
                    #  edges    = np.linspace(0.0, 12.001, 601),
                    xorders   = 4,
                    yorder   = 5,
                    ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v03', major='', inactive=mode_yb),
                        rounding = 3,
                        edges = np.concatenate(( [0.7], np.arange(1, 8.001, 0.02), [9.0, 12.0] )),
                        name = 'rebin',
                        label = 'Final histogram {detector}'
                        ),
                kinint2_enu = NestedDict(
                    bundle   = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2'), inactive=not mode_yb),
                    variables = ('enu_in', 'ctheta'),
                    edges     = np.linspace(1.8, 8.0, 601),
                    xorders   = 4,
                    yorder   = 5,
                    ),
                rebin_yb = NestedDict(
                        bundle = dict(name='rebin', version='v03', major='', inactive=not mode_yb),
                        rounding = 3,
                        edges = np.linspace(1.8, 8.0, 201),
                        name = 'rebin',
                        label = 'Final histogram {detector}'
                        ),
                ibd_xsec = NestedDict(
                    bundle = dict(name='xsec_ibd', version='v02'),
                    order = 1,
                    ),
                oscprob = NestedDict(
                    bundle = dict(name='oscprob', version='v03', major='rdc'),
                    pdgyear = self.opts.pdgyear
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
                eres = NestedDict(
                        bundle = dict(name='detector_eres_normal', version='v01', major='', inactive='multieres' in self.opts.energy_model),
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                        pars = uncertaindict([
                            ('a', (0.000, 'fixed')),
                            ('b', (0.03, 'fixed')),
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
                        correlations = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200/corrmap_xuyu.txt'
                        ),
                multieres = NestedDict(
                        bundle = dict(name='detector_multieres_stats', version='v01', major='s', inactive='multieres' not in self.opts.energy_model),
                        # pars: sigma_e/e = sqrt(b^2/E),
                        parameter = 'eres',
                        nph = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200/subdetector200_nph.txt',
                        expose_matrix = False
                        ),
                shape_uncertainty = NestedDict(
                        unc = uncertain(1.0, 1.0, 'percent'),
                        nbins = 200 # number of bins, the uncertainty is defined to
                        )
                )

        if mode_yb:
            from physlib import PhysicsConstants
            pc = PhysicsConstants(2016)
            shift = pc.DeltaNP - pc.ElectronMass
            histshift = R.HistEdgesLinear(1.0, -shift)
            self.cfg.enuToEvis0 = NestedDict(
                bundle  = dict(name='predefined', version='v01'),
                name    = 'enuToEvis0',
                inputs  = (histshift.histedges.hist_in,),
                outputs = histshift.histedges.hist,
                object  = histshift
                )
        if not 'lsnl' in self.opts.correlation:
            self.cfg.lsnl.correlations = None
            self.cfg.lsnl.correlations_pars = None
        if not 'subdetectors' in self.opts.correlation:
            self.cfg.subdetector_fraction.correlations = None

        if self.opts.parameters in ['yb', 'yb-noosc']:
            self.cfg.eff.pars = uncertain(0.73, 'fixed')
            self.cfg.livetime.pars['AD1'] = uncertain( 6*330*seconds_per_day, 'fixed' )

    def preinit_variables(self):
        mode_yb = self.opts.mode.startswith('yb')

        if self.opts.spectrum_unc in ['final', 'initial']:
            spec = self.namespace('spectrum')
            cfg = self.cfg.shape_uncertainty
            unc = cfg.unc

            if self.opts.spectrum_unc=='initial':
                if mode_yb:
                    edges = self.cfg.kinint2_enu.edges
                else:
                    edges = self.cfg.kinint2.edges
            elif self.opts.spectrum_unc=='final':
                if mode_yb:
                    edges = self.cfg.rebin_yb.edges
                else:
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
        if self.opts.mode=='yb':
            ns.addobservable("Enu",    outputs.enu_in, export=False)
        else:
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
    formula_enu_yb = ['enu_in()', 'evis_hist = enuToEvis0(enu_in_hist())']

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
                                (sum[i]|  power_livetime_factor*anuspec[i](enu()))*
                                sum[c]|
                                  pmns[c]*oscprob[c,d,r](enu())
                            )
            '''

    formula_ibd_noeffects_yb = '''
                            kinint2(
                              sum[r]|
                                baselineweight[r,d]*
                                ibd_xsec(enu_in_mesh(), ctheta())*
                                (sum[i]|  power_livetime_factor*anuspec[i](enu_in_mesh()))*
                                sum[c]|
                                  pmns[c]*oscprob[c,d,r](enu_in_mesh())
                            )
            '''

    formula_back = 'observation=norm * rebin(ibd)'


    lib = dict(
            cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                           label='anu count rate | {isotope}@{reactor}-\\>{detector} ({component})'),
            cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
            cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),

            eres_weighted           = dict(expr='subdetector_fraction*eres', label='{{Fractional observed spectrum {subdetector}|weight: {weight_label}}}'),
            ibd                     = dict(expr=('eres', 'sum:c|eres_weighted'), label='Observed IBD spectrum | {detector}'),
            ibd_noeffects           = dict(expr='kinint2', label='Observed IBD spectrum (no effects) | {detector}'),
            ibd_noeffects_bf        = dict(expr='kinint2*shape_norm', label='Observed IBD spectrum (best fit, no effects) | {detector}'),

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

            countrate_rd            = dict(expr=('anuspec_rd*ibd_xsec*jacobian*oscprob_full', 'anuspec_rd*ibd_xsec*oscprob_full'), label='Countrate {reactor}-\\>{detector}'),
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
