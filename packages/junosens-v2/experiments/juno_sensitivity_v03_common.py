
from __future__ import print_function
from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna import constructors as C
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R
from collections import OrderedDict

class exp(baseexp):
    """
JUNO experiment implementation v03 (common) -> current

This version is using common inputs

Derived from:
    - [2019.12] Daya Bay model from dybOscar and GNA
    - [2019.12] juno_chengyp
    - [2020.04] juno_sensitivity_v01
    - [2020.06] juno_sensitivity_v02

Changes since previous implementation [juno_sensitivity_v02]:
    - Switch to dm32 (usin scan minimizer)
    - WIP: geo-neutrino
    - Common inputs from
      * Reactors
        + Power, no TS3/TS4
        + WIP: Power, HZ factor ((max(y-2,0) + max(y-3,0))/(y*2)) - not variable
        + Duty cycle: 11/12
      * DAQ
        + 6 years
        + # 8 years (6 times no TS3/4)
      * Oscillation parameters:
        + Now configured via the bundle, not physlib

Implements:
    - Reactor antineutrino flux:
      * Spectra:
        + Huber+Mueller
        + Free
      * SNF contribution
      * Off-equilibrium corrections (Mueller)
    - Vacuum 3nu oscillations
    - Evis mode with 2d integration (similary to dybOscar)
    - LSNL
    - Detector energy resolution

"""

    detectorname = 'AD1'

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')

        # Energy model
        emodel = parser.add_argument_group('emodel', description='Energy model parameters')
        emodel.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres'], default=['lsnl', 'eres'], help='Energy model components')

        # Backgrounds and geo-neutrino
        bkg=parser.add_argument_group('bkg', description='Background and geo-neutrino parameters')
        bkg_choices = ['acc', 'lihe', 'fastn', 'alphan', 'geonu']
        bkg.add_argument('-b', '--bkg', nargs='*', default=bkg_choices, choices=bkg_choices, help='Enable background group')

        # Binning
        binning=parser.add_argument_group('binning', description='Binning related options')
        binning.add_argument('--final-emin', type=float, help='Final binning Emin')
        binning.add_argument('--final-emax', type=float, help='Final binning Emax')

        # osc prob
        parser.add_argument('--oscprob', choices=['vacuum'], default='vacuum', help='oscillation probability type')
        # parser.add_argument('--oscprob', choices=['vacuum', 'matter'], default='vacuum', help='oscillation probability type')

        # Parameters
        parser.add_argument('--pdgyear', choices=[2016, 2018], default=2018, type=int, help='PDG version to read the oscillation parameters')
        parser.add_argument('--spectrum-unc', action='store_true', help='type of the spectral uncertainty')
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
        if 'multieres' in self.opts.energy_model:
            self.subdetectors_names = ['subdet%03i'%i for i in range(5)]
        else:
            self.subdetectors_names = ()
        self.reactors = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'DYB', 'HZ']

        self.nidx = [
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     self.reactors],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('s', 'subdetector', self.subdetectors_names),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] ),
        ]
        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        energy_model = self.opts.energy_model
        if 'eres' in energy_model and 'multieres' in energy_model:
            raise Exception('Energy model options "eres" and "multieres" are mutually exclusive: use only one of them')

        #
        # Optional formula parts
        #
        formula_options = dict(
                #
                # Oscillation probability
                #
                oscprob = dict(
                    vacuum='sum[c]| pmns[c]*oscprob[c,d,r](enu())' ,
                    matter='oscprob_matter[d,r](enu())'
                    )[self.opts.oscprob],
                #
                # Geo neutrino
                #
                # geonu_spectrum = '+geonu_scale*bracket(geonu_norm_U238*geonu_spectrum_U238(enu()) + geonu_norm_Th232*geonu_spectrum_Th232(enu()))'
                                 # if 'geo' in self.opts.bkg else '',
                #
                # Reactor
                #
                shape_norm       = '*shape_norm()'                 if self.opts.spectrum_unc else '',
                #
                # Energy model
                #
                lsnl = 'lsnl|'                                                if 'lsnl' in energy_model else '',
                eres = 'eres|'                                                if 'eres' in energy_model else
                       'concat[s]| rebin| subdetector_fraction[s] * eres[s]|' if 'multieres' in energy_model else
                       '',
                #
                # Backgrounds
                #
                accidentals = '+accidentals' if 'acc'    in self.opts.bkg else '',
                lihe        = '+lihe'        if 'lihe'   in self.opts.bkg else '',
                alphan      = '+alphan'      if 'alphan' in self.opts.bkg else '',
                fastn       = '+fastn'       if 'fastn'  in self.opts.bkg else '',
                geonu       = '+geonu'       if 'geonu'  in self.opts.bkg else '',
                #
                # IBD rebinning
                #
                ibd         = 'ibd'          if 'multieres' in energy_model else 'rebin(ibd)',
                )

        self.formula = [
                # Some common definitions
                'baseline[d,r]',
                'livetime=bracket(daq_years*seconds_in_year)',
                'conversion_factor',
                'efflivetime = bracket(eff * livetime)',
                # 'geonu_scale = eff * livetime[d] * conversion_factor * target_protons',
                #
                # Neutrino energy
                #
                'evis_hist=evis_hist()',
                'enu| ee(evis()), ctheta()',
                #
                # Energy model
                #
                'lsnl_edges| evis_hist, evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()'if 'lsnl' in energy_model else '',
                'eres_matrix| evis_hist'     if 'eres'      in energy_model else '',
                #
                # Reactor part
                #
                'numerator = global_norm * efflivetime * duty_cycle * thermal_power_scale[r] * thermal_power_nominal[r] * '
                             'fission_fractions_scale[r,i] * fission_fractions_nominal[r,i]() * '
                             'conversion_factor * target_protons',
                'isotope_weight = energy_per_fission_scale[i] * energy_per_fission[i] * fission_fractions_scale[r,i]',
                'energy_per_fission_avg = sum[i]| isotope_weight * fission_fractions_nominal[r,i]()',
                'power_livetime_factor = numerator / energy_per_fission_avg',
                'anuspec[i](enu())',
                #
                # SNF
                #
                'energy_per_fission_avg_nominal = sum[i] | energy_per_fission[i] * fission_fractions_nominal[r,i]()',
                'snf_plf_daily = conversion_factor * duty_cycle * thermal_power_nominal[r] * fission_fractions_nominal[r,i]() / energy_per_fission_avg_nominal',
                'nominal_spec_per_reac =  sum[i]| snf_plf_daily*anuspec[i]()',
                'snf_in_reac = snf_norm * efflivetime * target_protons * snf_correction(enu(), nominal_spec_per_reac)',
                #
                # Backgrounds
                #
                'accidentals = bracket| days_in_second * efflivetime * acc_rate    * acc_rate_norm    * rebin_acc[d]   | acc_spectrum()',
                'fastn       = bracket| days_in_second * efflivetime * fastn_rate  * fastn_rate_norm  * rebin_fastn[d] | fastn_spectrum()',
                'alphan      = bracket| days_in_second * efflivetime * alphan_rate * alphan_rate_norm * rebin_alphan[d]| alphan_spectrum()',
                'lihe        = bracket| days_in_second * efflivetime * lihe_rate   * lihe_rate_norm   * rebin_lihe[d]  | lihe_spectrum()',
                'geonu       = bracket| days_in_second * efflivetime * geonu_rate  * geonu_rate_norm  * rebin_geonu[d] | frac_Th232 * geonu_Th232_spectrum() + frac_U238 * geonu_U238_spectrum()',
                'bkg_shape_variance = bin_width_factor * bkgbin_widths(accidentals) * sumsq_snapshot| sumsq_bkg| lihe_bin2bin*lihe, fastn_bin2bin*fastn, alphan_bin2bin*alphan',
                #
                # IBD part
                #
                '''ibd={eres} {lsnl}
                    kinint2(
                      sum[r]|
                        (
                            baselineweight[r,d]*
                            ( reactor_active_norm * (sum[i]| power_livetime_factor*offeq_correction[i,r](enu(), anuspec[i]()) ) + snf_in_reac )*
                            {oscprob}
                        )*
                        bracket(
                            ibd_xsec(enu(), ctheta())*
                            jacobian(enu(), ee(), ctheta())
                        )
                    ) {shape_norm}
                '''.format(**formula_options),
                #
                # Total observation
                #
                'observation=norm*{ibd} {accidentals} {lihe} {alphan} {fastn} {geonu}'.format(**formula_options),
                'juno_variance = staterr2(observation) + bkg_shape_variance'
                ]

    def parameters(self):
        ns = self.namespace
        dmxx = 'pmns.DeltaMSq23'
        for par in [dmxx, 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']:
            ns[par].setFree()

    def init_configuration(self):
        edges    = np.arange(0.0, 12.001, 0.01)
        edges_final = np.concatenate( (
                                    [0.8],
                                    np.arange(1, 6.0, 0.02),
                                    np.arange(6, 7.0, 0.1),
                                    [7.0, 7.5, 12.0]
                                )
                            )
        if self.opts.final_emin is not None:
            print('Truncate final binning E>={}'.format(self.opts.final_emin))
            edges_final = edges_final[edges_final>=self.opts.final_emin]
        if self.opts.final_emax is not None:
            print('Truncate final binning E<={}'.format(self.opts.final_emax))
            edges_final = edges_final[edges_final<=self.opts.final_emax]
        if self.opts.final_emin is not None or self.opts.final_emax is not None:
            print('Final binning:', edges_final)

        self.cfg = NestedDict(
                #
                # Numbers
                #
                numbers0 = OrderedDict(
                    bundle = dict(name='parameters', version='v05'),
                    state='fixed',
                    labels=dict(
                        seconds_in_year   = 'Number of seconds in year',
                        days_in_second    = 'Number of days in a second',
                        conversion_factor = 'Conversion factor: W[GW]/[MeV] N[fissions]->N[fissions]/T[s]',
                        ),
                    pars =  dict(
                            seconds_in_year   = 365.0*24.0*60.0*60.0,
                            days_in_second    = 1.0/(24.0*60.0*60.0),
                            conversion_factor = R.NeutrinoUnits.reactorPowerConversion, #taken from transformations/neutrino/ReactorNorm.cc
                            ),
                    ),
                numbers = OrderedDict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/juno_input_numbers.py',
                    skip = ('percent',),
                    state= 'fixed'
                    ),
                bkg_shape_unc = OrderedDict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/bkg_bin2bin.yaml',
                    hooks = OrderedDict(bin_width_factor=lambda pars: (1.0/pars['bin_width'], '1/`bin width`')),
                    state= 'fixed'
                    ),
                # Detector and reactor
                norm = OrderedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "norm",
                        label = 'Reactor power/detection efficiency correlated normalization',
                        pars = uncertain(1.0, (2**2+1**2)**0.5, 'percent')
                        ),
                baselines = OrderedDict(
                        bundle = dict(name='reactor_baselines', version='v02', major = 'rd'),
                        reactors  = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/reactor_baselines.yaml',
                        reactors_key = 'reactor_baseline',
                        detectors = dict(AD1=0.0),
                        unit = 'km'
                        ),
                # Reactor
                energy_per_fission =  OrderedDict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/energy_per_fission.yaml',
                        separate_uncertainty = '{}_scale'
                        ),
                thermal_power =  OrderedDict(
                        bundle = dict(name="parameters", version = "v06", names=dict(thermal_power='thermal_power_nominal')),
                        pars = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/thermal_power.yaml',
                        separate_uncertainty = '{}_scale'
                        ),
                # Backgrounds
                bkg_rate = OrderedDict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/bkg_rates.yaml',
                        separate_uncertainty = '{}_norm'
                    ),
                #
                # Transformations
                #
                # General
                kinint2 = OrderedDict(
                    bundle    = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2')),
                    variables = ('evis', 'ctheta'),
                    edges     = edges,
                    #  edges   = np.linspace(0.0, 12.001, 601),
                    xorders   = 4,
                    yorder    = 5,
                    ),
                rebin = OrderedDict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = edges_final,
                    instances={
                        'rebin': 'Final histogram {detector}',
                        }
                    ),
                rebin_bkg = OrderedDict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = edges_final,
                    instances={
                        'rebin_acc': 'Accidentals {autoindex}',
                        'rebin_lihe': '9Li/8He {autoindex}',
                        'rebin_fastn': 'Fast neutrons {autoindex}',
                        'rebin_alphan': 'C(alpha,n)O {autoindex}',
                        'rebin_geonu': 'Geo-nu {autoindex}'
                        }
                    ),
                bkg_shape_uncertainty = OrderedDict(
                    bundle = dict(name='trans_sumsq', version='v01', major=''),
                    ninputs = 3,
                    instances={'sumsq_bkg': 'Bkg shape variance {autoindex}'}
                    ),
                bkg_edges = OrderedDict(
                    bundle = dict(name='trans_histedges', version='v01', major=''),
                    types = ('widths', ),
                    instances={'bkgbin': 'Bkg bins {autoindex}'}
                    ),
                variance = OrderedDict(
                    bundle = dict(name='trans_snapshot', version='v01', major=''),
                    instances={'sumsq_snapshot': 'Bkg shape variance snapshot, not corrected|{autoindex}',
                               'staterr2': 'Stat. errors (snapshot)'}
                    ),
                # Oscillations and detection
                ibd_xsec = OrderedDict(
                        bundle = dict(name='xsec_ibd', version='v02'),
                        order = 1,
                        ),
                oscprob = OrderedDict(
                        bundle = dict(name='oscprob', version='v05', major='rdc', inactive=self.opts.oscprob=='matter'),
                        parameters = dict(
                            DeltaMSq23    = 2.453e-03,
                            DeltaMSq12    = 7.53e-05,
                            SinSqDouble13 = (0.08529904, 0.00267792),
                            SinSqDouble12 = 0.851004,
                            # SinSq13 = (0.0218, 0.0007),
                            # SinSq12 = 0.307,
                            )
                        ),
                # oscprob_matter = OrderedDict(
                    # bundle = dict(name='oscprob_matter', version='v01', major='rd', inactive=self.opts.oscprob=='vacuum',
                                  # names=dict(oscprob='oscprob_matter')),
                    # density = 2.6, # g/cm3
                    # pdgyear = self.opts.pdgyear,
                    # dm      = '23'
                    # ),
                anuspec_hm = OrderedDict(
                        bundle = dict(name='reactor_anu_spectra', version='v05'),
                        name = 'anuspec',
                        filename = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/deprecated/JUNOInputs2020_6_26.root',
                        objectnamefmt = 'HuberMuellerFlux_{isotope}',
                        free_params=False, # enable free spectral model
                        varmode='log',
                        varname='anue_weight_{index:02d}',
                        ns_name='spectral_weights',
                        edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.025 ), [ 12.3 ] ) ),
                        ),
                offeq_correction = OrderedDict(
                        bundle = dict(name='reactor_offeq_spectra', version='v05', major=''),
                        offeq_data = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/deprecated/JUNOInputs2020_6_26.root',
                        objectnamefmt = 'NonEq_FluxRatio',
                        relsigma = 0.3,
                        ),
                fission_fractions = OrderedDict(
                        bundle = dict(name="parameters_yaml_v01", major = 'i'),
                        parameter = 'fission_fractions_nominal',
                        separate_uncertainty = "fission_fractions_scale",
                        label = 'Fission fraction of {isotope} in reactor {reactor}',
                        objectize=True,
                        data = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/files/fission_fractions.yaml'
                        ),
                snf_correction = OrderedDict(
                        bundle = dict(name='reactor_snf_spectra', version='v05', major='r'),
                        snf_average_spectra = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/deprecated/JUNOInputs2020_6_26.root',
                        objectname = 'SNF_FluxRatio'
                        ),
                lsnl = OrderedDict(
                        bundle     = dict(name='energy_nonlinearity_db_root', version='v03', major='l'),
                        names      = OrderedDict( [
                            ('nominal', 'positronScintNL'),
                            ('pull0', 'positronScintNLpull0'),
                            ('pull1', 'positronScintNLpull1'),
                            ('pull2', 'positronScintNLpull2'),
                            ('pull3', 'positronScintNLpull3'),
                            ]),
                        filename   = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/deprecated/JUNOInputs2020_6_26.root',
                        edges      = 'evis_edges',
                        extrapolation_strategy = 'extrapolate',
                        nonlin_range = (0.95, 12.),
                        expose_matrix = True
                        ),
                shape_uncertainty = OrderedDict(
                        unc = uncertain(1.0, 1.0, 'percent'),
                        nbins = 200 # number of bins, the uncertainty is defined to
                        ),
                snf_norm = OrderedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'snf_norm',
                        label='SNF norm',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                reactor_active_norm = OrderedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'reactor_active_norm',
                        label='Reactor nu (active norm)',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                #
                # Backgrounds
                #
                bkg_spectra = OrderedDict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = 'data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input/deprecated/JUNOInputs2020_6_26.root',
                    formats = ['AccBkgHistogramAD',           'Li9BkgHistogramAD',       'FnBkgHistogramAD',       'AlphaNBkgHistogramAD',   'GeoNuTh232'                 , 'GeoNuU238'],
                    names   = ['acc_spectrum',                'lihe_spectrum',           'fastn_spectrum',         'alphan_spectrum',        'geonu_Th232_spectrum'       , 'geonu_U238_spectrum'],
                    labels  = ['Accidentals|(norm spectrum)', '9Li/8He|(norm spectrum)', 'Fast n|(norm spectrum)', 'AlphaN|(norm spectrum)', 'GeoNu Th232|(norm spectrum)', 'GeoNu U238|(norm spectrum)'],
                    normalize = True,
                    ),
                geonu_fractions=OrderedDict(
                        bundle = dict(name='var_fractions_v02'),
                        names = [ 'Th232', 'U238' ],
                        format = 'frac_{component}',
                        fractions = uncertaindict(
                            Th232 = ( 0.23, 'fixed' )
                            ),
                        ),
                )

        if 'eres' in self.opts.energy_model:
            self.cfg.eres = OrderedDict(
                    bundle = dict(name='detector_eres_normal', version='v01', major=''),
                    # pars: sigma_e/e = sqrt(a^2 + b^2/E + c^2/E^2),
                    # a - non-uniformity
                    # b - statistical term
                    # c - noise
                    parameter = 'eres',
                    pars = uncertaindict([
                        ('a_nonuniform', (0.0082, 0.0001)),
                        ('b_stat',       (0.0261, 0.0002)),
                        ('c_noise',      (0.0123, 0.0004))
                        ],
                        mode='absolute'),
                    expose_matrix = False
                    )
        elif 'multieres' in self.opts.energy_model:
            self.cfg.subdetector_fraction = OrderedDict(
                    bundle = dict(name="parameters", version = "v03"),
                    parameter = "subdetector_fraction",
                    label = 'Subdetector fraction weight for {subdetector}',
                    pars = uncertaindict(
                        [(subdet_name, (0.2, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
                        ),
                    covariance = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_cov.txt'
                    )
            self.cfg.multieres = OrderedDict(
                    bundle = dict(name='detector_multieres_stats', version='v01', major='s'),
                    # pars: sigma_e/e = sqrt(b^2/E),
                    parameter = 'eres',
                    relsigma = self.opts.eres_b_relsigma,
                    nph = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_nph.txt',
                    rescale_nph = self.opts.eres_npe,
                    expose_matrix = False
                    )

        if not 'lsnl' in self.opts.correlation:
            self.cfg.lsnl.correlations = None
            self.cfg.lsnl.correlations_pars = None
        if not 'subdetectors' in self.opts.correlation:
            self.cfg.subdetector_fraction.correlations = None

    def preinit_variables(self):
        if self.opts.spectrum_unc:
            spec = self.namespace('spectrum')
            cfg = self.cfg.shape_uncertainty
            unc = cfg.unc
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

            self.cfg.shape_uncertainty = OrderedDict(
                    bundle = dict(name='predefined', version='v01'),
                    name   = 'shape_norm',
                    inputs = None,
                    outputs = vararray.single(),
                    unc    = cfg.unc,
                    object = vararray
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

        if self.opts.verbose>2:
            print('Expression tree:')
            self.expression.dump_all(True)
            print()
        elif self.opts.verbose>1:
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
            self.namespace.printparameters(labels=55, stats=self.stats, correlations=correlations)

    def register(self):
        from gna.env import env
        futurens = env.future.child(('spectra', self.namespace.name))

        outputs = self.context.outputs

        if 'ibd_noeffects_bf' in outputs:
            fine = outputs.ibd_noeffects_bf.AD1
        else:
            fine = outputs.kinint2.AD1

        if 'lsnl' in self.opts.energy_model:
            fine = outputs.lsnl.AD1

        if 'eres' in self.opts.energy_model:
            fine = outputs.eres.AD1

        if 'multieres' in self.opts.energy_model:
            fine = outputs.ibd.AD1

        futurens[('variance', self.detectorname, 'stat')]     = outputs.staterr2.AD1
        futurens[('variance', self.detectorname, 'bkgshape')] = outputs.bkg_shape_variance.AD1
        futurens[('variance', self.detectorname, 'full')]     = outputs.juno_variance.AD1
        # Force calculation of the stat errors
        outputs.juno_variance.AD1.data()

        futurens[(self.detectorname, 'initial')] = outputs.kinint2.AD1
        futurens[(self.detectorname, 'fine')] = fine
        futurens[(self.detectorname, 'final')] = outputs.observation.AD1
        if 'lsnl' in self.opts.energy_model:
            futurens[(self.detectorname, 'lsnl')] = outputs.lsnl.AD1

        if 'eres' in self.opts.energy_model:
            futurens[(self.detectorname, 'eres')] = outputs.eres.AD1

        k0 = ('extra',)
        for k, v in self.context.outputs.items(nested=True):
            futurens[k0+k]=v

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.rebin.AD1
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    lib = """
        #
        # Control
        #
        livetime:
            expr: 'daq_years*seconds_in_year'
            label: JUNO livetime, s
        #
        # Reactor part
        #
        powerlivetime_factor:
            expr: 'conversion_factor*duty_cycle*efflivetime*fission_fractions_scale*global_norm*target_protons*thermal_power_nominal*thermal_power_scale'
            label: 'Power/Livetime/Mass factor, nominal'
        power_factor_snf:
            expr: 'conversion_factor*duty_cycle*thermal_power_nominal'
            label: 'Power factor for SNF'
        livetime_factor_snf:
            expr: 'efflivetime*snf_norm*target_protons'
            label: 'Livetime/mass factor for SNF, b.fit'
        baselinewight_switch:
            expr: 'baselineweight*reactor_active_norm'
            label: 'Baselineweight (toggle) for {reactor}-\\>{detector}'
        energy_per_fission_fraction:
            expr: 'fission_fractions_nominal*isotope_weight'
            label: Fractional energy per fission
        energy_per_fission_avg:
          expr: 'energy_per_fission_avg'
          label: 'Average energy per fission at {reactor}'
        #
        # Reactor spectrum
        #
        reac_spectrum_oscillated:
          expr:
          - 'anuspec_rd*oscprob_full'
          - 'anuspec_rd_full*oscprob_full'
          label: 'Reactor spectrum osc. {reactor}-\\>{detector}'
        anuspec_rd:
          expr:
          - sum:i|anuspec_weighted
          - sum:i|anuspec_weighted_offeq
          - sum:i|anuspec_equilib_ri
          label: '{{Antineutrino spectrum|{reactor}}}'
        anuspec_rd_switch:
          expr: 'anuspec_rd*reactor_active_norm'
          label: '{{Antineutrino spectrum|{reactor}}}'
        anuspec_rd_full:
          expr: 'anuspec_rd_switch+snf_in_reac'
          label: '{{Antineutrino spectrum+SNF|{reactor}}}'
        anuspec_weighted:
          expr: 'anuspec*power_livetime_factor'
          label: '{{Antineutrino spectrum|{reactor}.{isotope}-\\>{detector}}}'
        anuspec_weighted_offeq:
          expr:
          - anuspec*offeq_correction*power_livetime_factor
          - offeq_correction*power_livetime_factor
          label: '{{Antineutrino spectrum (+offeq)|{reactor}.{isotope}}}'
        #
        # Backgrounds
        #
        acc_num:
            expr: 'acc_rate*acc_rate_norm*days_in_second*efflivetime'
            label: Number of accidentals (b.fit)
        fastn_num:
            expr: 'days_in_second*efflivetime*fastn_rate*fastn_rate_norm'
            label: Number of fast neutrons (b.fit)
        alphan_num:
            expr: 'alphan_rate*alphan_rate_norm*days_in_second*efflivetime'
            label: Number of alpha-n (b.fit)
        lihe_num:
            expr: 'days_in_second*efflivetime*lihe_rate*lihe_rate_norm'
            label: Number of 9Li/8He (b.fit)
        geonu_num:
            expr: 'days_in_second*efflivetime*geonu_rate*geonu_rate_norm'
            label: Number of Geo nu events (b.fit)
        geonu_Th232_spectrum_tot:
            expr: 'frac_Th232*geonu_Th232_spectrum'
            label: 'Geonu Th232 total spectrum'
        geonu_U238_spectrum_tot:
            expr: 'frac_U238*geonu_U238_spectrum'
            label: 'Geonu U238 total spectrum'
        geonu_spectrum:
            expr: geonu_Th232_spectrum_tot+geonu_U238_spectrum_tot
            label: Total geo nu spectrum
        #
        # Oscillation probability
        #
        oscprob_weighted:
          expr: 'oscprob*pmns'
        oscprob_full:
          expr: 'sum:c|oscprob_weighted'
          label: 'anue survival probability|{reactor}-\\>{detector}|weight: {weight_label}'
        #
        # Spectrum and oscillations
        #
        cspec_diff:
          expr: 'anuspec*ibd_xsec*jacobian*oscprob'
          label: 'anu count rate | {isotope}@{reactor}-\\>{detector} ({component})'
        cspec_diff_reac_l:
          expr: 'baselineweight*cspec_diff_reac'
        cspec_diff_det_weighted:
          expr: 'pmns*cspec_diff_det'
        #
        # Detector stage
        #
        reac_spectrum_at_detector:
          expr:
          - 'baselineweight_switch*reac_spectrum_oscillated'
          - 'baselineweight*reac_spectrum_oscillated'
          label: '{reactor} spectrum at {detector}'
        observable_spectrum_reac:
          expr: 'ibd_xsec_rescaled*reac_spectrum_at_detector'
          label: 'Observable spectrum from {reactor} at {detector}'
        observable_spectrum:
          expr: 'sum:r|observable_spectrum_reac'
          label: 'Observable spectrum at {detector}'
        observation_ibd:
          expr: norm*rebin
          label: 'Observable spectrum at JUNO'
        #
        # Others
        #
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
        fission_fractions:
          expr: 'fission_fractions[r,i]()'
          label: "Fission fraction for {isotope} at {reactor}"
        energy_per_fission_weight:
          expr: 'energy_per_fission_weight'
          label: "Weighted energy_per_fission for {isotope} at {reactor}"
        energy_per_fission_weighted:
          expr: 'energy_per_fission*fission_fractions'
          label: "{{Energy per fission for {isotope} | weighted with fission fraction at {reactor}}}"
        power_livetime_factor:
          expr: 'power_livetime_factor'
          label: '{{Power-livetime factor (~nu/s)|{reactor}.{isotope}}}'
        numerator:
          expr: 'numerator'
          label: '{{Power-livetime factor (~MW)|{reactor}.{isotope}}}'
        power_livetime_scale:
          expr: 'eff*livetime*thermal_power_scale*thermal_power_nominal*conversion_factor*target_protons'
          label: '{{Power-livetime factor (~MW)| {reactor}.{isotope}}}'
        countrate_rd:
          expr:
          - 'anuspec_rd*ibd_xsec*jacobian*oscprob_full'
          - 'anuspec_rd*ibd_xsec*oscprob_full'
          label: 'Countrate {reactor}-\\>{detector}'
        ibd_xsec_rescaled:
          expr: 'ibd_xsec*jacobian'
          label: IBD cross section vs Evis
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
