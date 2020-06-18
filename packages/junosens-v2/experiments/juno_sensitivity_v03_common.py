# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna import constructors as C
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R

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

Implements:
    - Reactor antineutrino flux:
      * Spectra:
        + ILL+Vogel (now default)
        + Huber+Mueller
        + Free
      * [optional] Off-equilibrium corrections (Mueller)
      * [optional] SNF contribution
    - Vacuum 3nu oscillations
    - Evis mode with 2d integration (similary to dybOscar)
    - Final binning:
      * 20 keV
      * 10 keV (default)
    - [optional] Birks-Cherenkov detector energy responce (Yaping)
    - [optional] Detector energy resolution
    - [optional] Multi-detector energy resolution (Yaping), concatenated

Misc changes:
    - Switch oscillation probability bundle from v03 to v04 (OscProb3 class)
    - Switch to double angle parameters for theta12 and theta13
    - Added concatenated subdetectors
    - Uncomment uncertainties:
      * energy per fission
      * fission fractions
    - [2020.04.24] Modify fluxes summation and integration sequence to introduce geo-neutrino spectra
    - [2020.05.05] Add backgrounds
    - [2020.05.11] Remove multieres summation mode
    - [2020.05.11] Add SNF
    - [2020.05.15] Remove multieres option. Keep the code (not working)
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
        emodel = parser.add_argument_group('emodel', description='Energy model parameters')
        emodel.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres'], default=['lsnl', 'eres'], help='Energy model components')
        emodel.add_argument('--eres-b-relsigma', type=float, help='Energy resolution parameter (b) relative uncertainty')

        eres = emodel.add_mutually_exclusive_group()
        eres.add_argument('--eres-sigma', type=float, help='Energy resolution at 1 MeV')
        eres.add_argument('--eres-npe', type=float, default=1350.0, help='Average Npe at 1 MeV')

        # Backgrounds and geo-neutrino
        bkg=parser.add_argument_group('bkg', description='Background and geo-neutrino parameters')
        bkg_choices = ['acc', 'lihe', 'fastn', 'alphan']
        bkg.add_argument('-b', '--bkg', nargs='*', default=[], choices=bkg_choices, help='Enable background group')

        # Binning
        binning=parser.add_argument_group('binning', description='Binning related options')
        binning.add_argument('--estep', default=0.01, choices=[0.02, 0.01], type=float, help='Internal binning step')
        binning.add_argument('--final-emin', type=float, help='Final binning Emin')
        binning.add_argument('--final-emax', type=float, help='Final binning Emax')

        # reactor flux
        parser.add_argument('--flux', choices=['huber-mueller', 'ill-vogel'], default='ill-vogel', help='Antineutrino flux')
        parser.add_argument('--offequilibrium-corr', action='store_true', help="Turn on offequilibrium correction to antineutrino spectra")
        parser.add_argument('--snf', action='store_true', help="Enable SNF contribution")

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
            ('s', 'subdetector', self.subdetectors_names)
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
                geonu_spectrum = '+geonu_scale*bracket(geonu_norm_U238*geonu_spectrum_U238(enu()) + geonu_norm_Th232*geonu_spectrum_Th232(enu()))'
                                 if 'geo' in self.opts.bkg else '',
                #
                # Reactor
                #
                offeq_correction = '*offeq_correction[i,r](enu())' if self.opts.offequilibrium_corr else '',
                shape_norm       = '*shape_norm()'                 if self.opts.spectrum_unc else '',
                snf              = '+snf_in_reac'                  if self.opts.snf else '',
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
                # geonu       = '+geonu'       if 'geonu'  in self.opts.bkg else '',
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
                # 'geonu_scale = eff * livetime[d] * conversion_factor * target_protons[d]',
                #
                # Neutrino energy
                #
                'evis_hist=evis_hist()',
                'enu| ee(evis()), ctheta()',
                #
                # Energy model
                #
                'evis_edges_hist| evis_hist' if 'lsnl'      in energy_model else '',
                'eres_matrix| evis_hist'     if 'eres'      in energy_model else '',
                'eres_matrix[s]| evis_hist'  if 'multieres' in energy_model else '',
                #
                # Reactor part
                #
                'numerator = efflivetime * duty_cycle * thermal_power_scale[r] * thermal_power_nominal[r] * '
                             'fission_fractions_scale[r,i] * fission_fractions_nominal[r,i]() * '
                             'conversion_factor * target_protons[d]',
                'isotope_weight = eper_fission_scale[i] * eper_fission_nominal[i] * fission_fractions_scale[r,i]',
                'eper_fission_avg = sum[i]| isotope_weight * fission_fractions_nominal[r,i]()',
                'power_livetime_factor = numerator / eper_fission_avg',
                'anuspec[i](enu())',
                #
                # SNF
                #
                'eper_fission_avg_nominal = sum[i] | eper_fission_nominal[i] * fission_fractions_nominal[r,i]()',
                'snf_plf_daily = conversion_factor * duty_cycle * thermal_power_nominal[r] * fission_fractions_nominal[r,i]() / eper_fission_avg_nominal',
                'nominal_spec_per_reac =  sum[i]| snf_plf_daily*anuspec[i]()',
                'snf_in_reac = snf_norm * efflivetime * target_protons[d] * snf_correction(enu(), nominal_spec_per_reac)',
                #
                # Backgrounds
                #
                'accidentals = days_in_second * efflivetime * acc_rate    * acc_norm    * rebin_acc[d]|    acc_spectrum[d]()',
                'fastn       = days_in_second * efflivetime * fastn_rate  * fastn_norm  * rebin_fastn[d]|  fastn_spectrum[d]()',
                'alphan      = days_in_second * efflivetime * alphan_rate * alphan_norm * rebin_alphan[d]| alphan_spectrum[d]()',
                'lihe        = days_in_second * efflivetime * lihe_rate   * lihe_norm   * bracket| frac_li * rebin_li[d](li_spectrum()) + frac_he * rebin_he[d](he_spectrum())',
                #
                # IBD part
                #
                '''ibd={eres} {lsnl}
                    kinint2(
                      sum[r]|
                        (
                            baselineweight[r,d]*
                            ( reactor_active_norm * (sum[i] ( power_livetime_factor*anuspec[i](){offeq_correction}) ) {snf} )*
                            {oscprob}
                            {geonu_spectrum}
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
                'observation=norm*{ibd} {accidentals} {lihe} {alphan} {fastn}'.format(**formula_options)
                ]

    def parameters(self):
        ns = self.namespace
        dmxx = 'pmns.DeltaMSq23'
        for par in [dmxx, 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']:
            ns[par].setFree()

    def init_configuration(self):
        if self.opts.eres_npe:
            self.opts.eres_sigma = self.opts.eres_npe**-0.5
        else:
            self.opts.eres_npe = self.opts.eres_sigma**-2
        print('Energy resolution at 1 MeV: {}% ({} pe)'.format(self.opts.eres_sigma*100, self.opts.eres_npe))

        edges    = np.arange(0.0, 12.001, 0.01) #FIXME
        edges_final = np.concatenate( (
                                    [0.7],
                                    np.arange(1, 6.0, self.opts.estep),
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
                numbers = NestedDict(
                    bundle = dict(name='parameters', version='v04'),
                    labels=dict(
                        eff               = 'Detection efficiency',
                        daq_years         = 'Number of DAQ years',
                        duty_cycle        = 'Reactor duty cycle (per year)',
                        seconds_in_year   = 'Number of seconds in year',
                        conversion_factor = 'Conversion factor from GWt to MeV',
                        ),
                    pars = uncertaindict(
                        dict(
                            eff               = 0.82,
                            daq_years         = 6.0,
                            duty_cycle        = 11.0/12.0,
                            seconds_in_year   = 365.0*24.0*60.0*60.0,
                            conversion_factor = R.NeutrinoUnits.reactorPowerConversion, #taken from transformations/neutrino/ReactorNorm.cc
                            ),
                        mode='fixed')
                    ),
                kinint2 = NestedDict(
                    bundle    = dict(name='integral_2d1d', version='v03', names=dict(integral='kinint2')),
                    variables = ('evis', 'ctheta'),
                    edges     = edges,
                    #  edges   = np.linspace(0.0, 12.001, 601),
                    xorders   = 4,
                    yorder    = 5,
                    ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v04', major=''),
                        rounding = 5,
                        edges = edges_final,
                        instances={
                            'rebin': 'Final histogram {detector}',
                            }
                        ),
                rebin_bkg = NestedDict(
                        bundle = dict(name='rebin', version='v04', major=''),
                        rounding = 5,
                        edges = edges_final,
                        instances={
                            'rebin_acc': 'Accidentals {autoindex}',
                            'rebin_li': '9Li {autoindex}',
                            'rebin_he': '8He {autoindex}',
                            'rebin_fastn': 'Fast neutrons {autoindex}',
                            'rebin_alphan': 'C(alpha,n)O {autoindex}'
                            }
                        ),
                ibd_xsec = NestedDict(
                    bundle = dict(name='xsec_ibd', version='v02'),
                    order = 1,
                    ),
                oscprob = NestedDict(
                    bundle = dict(name='oscprob', version='v05', major='rdc', inactive=self.opts.oscprob=='matter'),
                    parameters = dict(
                        DeltaMSq23    = 2.444e-03,
                        DeltaMSq12    = 7.53e-05,
                        SinSqDouble13 = (0.08529904, 0.00267792),
                        SinSqDouble12 = 0.851004,
                        # SinSq13 = (0.0218, 0.0007),
                        # SinSq12 = 0.307,
                        )
                    ),
                # oscprob_matter = NestedDict(
                    # bundle = dict(name='oscprob_matter', version='v01', major='rd', inactive=self.opts.oscprob=='vacuum',
                                  # names=dict(oscprob='oscprob_matter')),
                    # density = 2.6, # g/cm3
                    # pdgyear = self.opts.pdgyear,
                    # dm      = '23'
                    # ),
                anuspec_hm = NestedDict(
                    bundle = dict(name='reactor_anu_spectra', version='v04', inactive=self.opts.flux!='huber-mueller'),
                    name = 'anuspec',
                    filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                                'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                    free_params=False, # enable free spectral model
                    varmode='log',
                    varname='anue_weight_{index:02d}',
                    ns_name='spectral_weights',
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.025 ), [ 12.3 ] ) ),
                    ),
                anuspec_ill = NestedDict(
                    bundle = dict(name='reactor_anu_spectra', version='v04', inactive=self.opts.flux!='ill-vogel'),
                    name = 'anuspec',
                    filename = ['data/reactor_anu_spectra/ILL/ILL_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                                'data/reactor_anu_spectra/Vogel/Vogel_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                    free_params=False, # enable free spectral model
                    varmode='log',
                    varname='anue_weight_{index:02d}',
                    ns_name='spectral_weights',
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.025 ), [ 12.3 ] ) ),
                    ),
                offeq_correction = NestedDict(
                    bundle = dict(name='reactor_offeq_spectra',
                                  version='v03', major='ir'),
                    offeq_data = 'data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
                    ),
                geonu = NestedDict(
                    bundle = dict(name='geoneutrino_spectrum', version='v01'),
                    data   = 'data/data-common/geo-neutrino/2006-sanshiro/geoneutrino-luminosity_{isotope}_truncated.knt'
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
                    parameter = 'fission_fractions_nominal',
                    separate_uncertainty = "fission_fractions_scale",
                    label = 'Fission fraction of {isotope} in reactor {reactor}',
                    objectize=True,
                    data = 'data/data_juno/fission_fractions/2013.12.05_xubo.yaml'
                    ),
                snf_correction = NestedDict(
                    bundle = dict(name='reactor_snf_spectra', version='v04', major='r'),
                    snf_average_spectra = './data/data-common/snf/2004.12-kopeikin/kopeikin_0412.044_spent_fuel_spectrum_smooth.dat',
                    ),
                baselines = NestedDict(
                        bundle = dict(name='reactor_baselines', version='v01', major = 'rd'),
                        reactors  = 'data/juno_nominal/coordinates_reactors.py',
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
                        parameter = "thermal_power_nominal",
                        label = 'Thermal power of {reactor} in GWt',
                        pars = uncertaindict([
                            ('TS1',  4.6),
                            ('TS2',  4.6),
                            ('TS3',  4.6), # inactive
                            ('TS4',  4.6), # inactive
                            ('YJ1',  2.9),
                            ('YJ2',  2.9),
                            ('YJ3',  2.9),
                            ('YJ4',  2.9),
                            ('YJ5',  2.9),
                            ('YJ6',  2.9),
                            ('DYB', 17.4),
                            ('HZ',  17.4), # corrected by factor
                            ],
                            uncertainty=0.8,
                            mode='percent'
                            ),
                        separate_uncertainty = 'thermal_power_scale'
                        ),
                target_protons = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "target_protons",
                        label = 'Number of protons in {detector}',
                        pars = uncertaindict(
                            [('AD1', (1.42e33, 'fixed'))],
                            ),
                        ),
                days_in_second =  NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter='days_in_second',
                        label='Number of days in a second',
                        pars = uncertain(1.0/(24.0*60.0*60.0), 'fixed'),
                        ),
                eper_fission =  NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'eper_fission_nominal',
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
                        separate_uncertainty = 'eper_fission_scale'
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
                                ("normalizationEnergy",   (11.99, 'fixed'))
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
                        ),
                snf_norm = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'snf_norm',
                        label='SNF norm',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                reactor_active_norm = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'reactor_active_norm',
                        label='Reactor nu (active norm)',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                #
                # Backgrounds
                #
                acc_spectrum_db = NestedDict(
                    bundle    = dict(name='root_histograms_v04', inactive=True),
                    filename  = 'data/data_juno/bkg/acc/2016_acc_dayabay_p15a/dayabay_acc_spectrum_p15a.root',
                    format    = 'accidentals',
                    name      = 'acc_spectrum',
                    label     = 'Accidentals|norm spectrum',
                    normalize = True
                    ),
                acc_spectrum = NestedDict(
                    bundle    = dict(name='root_histograms_v04'),
                    filename  = 'data/data_juno/bkg/acc/2019_acc_malyshkin/acc_bckg_FVcut.root',
                    format    = 'hAcc',
                    name      = 'acc_spectrum',
                    label     = 'Accidentals|norm spectrum',
                    normalize = slice(200,-1),
                    xscale    = 1.e-3,
                    ),
                fastn_spectrum=NestedDict(
                        bundle=dict(name='histogram_flat_v01'),
                        name='fastn_spectrum',
                        edges=edges,
                        label='Fast neutron|norm spectrum',
                        normalize=(0.7, 12.0),
                        ),
                li_spectrum=NestedDict(
                    bundle    = dict(name='root_histograms_v03'),
                    filename  = 'data/data_juno/bkg/lihe/2014_lihe_ochoa/toyli9spec_BCWmodel_v1_2400.root',
                    format    = 'h_eVisAllSmeared',
                    name      = 'li_spectrum',
                    label     = '9Li spectrum|norm',
                    normalize = True,
                    ),
                he_spectrum= NestedDict(
                    bundle    = dict(name='root_histograms_v03'),
                    filename  = 'data/data_juno/bkg/lihe/2014_lihe_ochoa/toyhe8spec_BCWmodel_v1_2400.root',
                    format    = 'h_eVisAllSmeared',
                    name      = 'he_spectrum',
                    label     = '8He spectrum|norm',
                    normalize = True,
                    ),
                alphan_spectrum = NestedDict(
                    bundle    = dict(name='root_histograms_v03'),
                    filename  = 'data/data_juno/bkg/alphan/2012_dayabay_alphan/P12B_alphan_2400.root',
                    format    = 'AD1',
                    name      = 'alphan_spectrum',
                    label     = 'C(alpha,n) spectrum|(norm)',
                    normalize = True,
                    ),
                lihe_fractions=NestedDict(
                        bundle = dict(name='var_fractions_v02'),
                        names = [ 'li', 'he' ],
                        format = 'frac_{component}',
                        fractions = uncertaindict(
                            li = ( 0.95, 0.05, 'relative' )
                            ),
                        ),
                # bkg_spectra = NestedDict(
                    # bundle    = dict(name='root_histograms_v05'),
                    # filename  = 'data/data_juno/bkg/group/2020-05-JUNO-YB/JunoBkg_evis_2400.root',
                    # formats    = ['AccBkgHistogramAD',           'Li9BkgHistogramAD',   'FnBkgHistogramAD',       'AlphaNBkgHistogramAD',   'GeoNuHistogramAD'],
                    # names      = ['acc_spectrum',                'lihe_spectrum',       'fn_spectrum',            'alphan_spectrum',        'geonu_spectrum'],
                    # labels     = ['Accidentals|(norm spectrum)', '9Li|(norm spectrum)', 'Fast n|(norm spectrum)', 'AlphaN|(norm spectrum)', 'GeoNu combined|(norm spectrum)'],
                    # normalize = True,
                    # ),
                acc_rate = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'acc_rate',
                        label='Acc rate',
                        pars = uncertain(0.9, 1, 'percent'),
                        separate_uncertainty='acc_norm',
                        ),
                fastn_rate = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'fastn_rate',
                        label='Fast n rate',
                        pars = uncertain(0.1, 100, 'percent'),
                        separate_uncertainty='fastn_norm',
                        ),
                lihe_rate = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'lihe_rate',
                        label='9Li/8He rate',
                        pars = uncertain(1.6, 20, 'percent'),
                        separate_uncertainty='lihe_norm',
                        ),
                alphan_rate = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'alphan_rate',
                        label='C(alpha,n)O rate',
                        pars = uncertain(0.05, 50, 'percent'),
                        separate_uncertainty='alphan_norm',
                        ),
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
            self.cfg.subdetector_fraction = NestedDict(
                    bundle = dict(name="parameters", version = "v03"),
                    parameter = "subdetector_fraction",
                    label = 'Subdetector fraction weight for {subdetector}',
                    pars = uncertaindict(
                        [(subdet_name, (0.2, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
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

            self.cfg.shape_uncertainty = NestedDict(
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
            self.namespace.printparameters(labels=55, stats=self.stats, correlations=correlations)

    def register(self):
        ns = self.namespace
        from gna.env import env
        futurens = env.future.child(('spectra', self.namespace.name))

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
            sns = ns('{}_sub'.format(self.detectorname))
            for i, out in enumerate(outputs.rebin.AD1.values()):
                sns.addobservable("sub{:02d}".format(i), out)
            ns.addobservable("{0}_eres".format(self.detectorname),     outputs.ibd.AD1)
            fine = outputs.ibd.AD1

        ns.addobservable("{0}_fine".format(self.detectorname),         fine)
        ns.addobservable("{0}".format(self.detectorname),              outputs.observation.AD1)

        futurens[(self.detectorname, 'fine')] = fine
        futurens[(self.detectorname, 'final')] = outputs.observation.AD1
        if 'lsnl' in self.opts.energy_model:
            futurens[(self.detectorname, 'lsnl')] = outputs.lsnl.AD1

        if 'eres' in self.opts.energy_model:
            futurens[(self.detectorname, 'eres')] = outputs.eres.AD1

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
        #
        # Reactor part
        #
        powerlivetime_factor:
            expr: 'conversion_factor*duty_cycle*efflivetime*fission_fractions_scale*target_protons*thermal_power_nominal*thermal_power_scale'
            label: 'Power/Livetime/Mass factor, nominal'
        power_factor_snf:
            expr: 'conversion_factor*duty_cycle*thermal_power_nominal'
            label: 'Power factor for SNF'
        livetime_factor_snf:
            expr: 'efflivetime*snf_norm*target_protons'
            label: 'Livetime/mass factor for SNF, b.fit'
        baselinewight_switch:
            expr: 'baselineweight*reactor_active_norm'
            label: 'Baselineweight (toggle)'
        eper_fission_fraction:
            expr: 'fission_fractions_nominal*isotope_weight'
            label: Fractional energy per fission
        eper_fission_avg:
          expr: 'eper_fission_avg'
          label: 'Average energy per fission at {reactor}'
        #
        # Backgrounds
        #
        acc_num:
            expr: 'acc_norm*acc_rate*days_in_second*efflivetime'
            label: Number of accidentals (b.fit)
        fastn_num:
            expr: 'days_in_second*efflivetime*fastn_norm*fastn_rate'
            label: Number of fast neutrons (b.fit)
        alphan_num:
            expr: 'alphan_norm*alphan_rate*days_in_second*efflivetime'
            label: Number of alpha-n (b.fit)
        lihe_num:
            expr: 'days_in_second*efflivetime*lihe_norm*lihe_rate'
            label: Number of 9Li/8He (b.fit)
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
        reac_spectrum_oscillated:
          expr: 'anuspec_rd*oscprob_full'
          label: 'Reactor spectrum osc. {reactor}-\\>{detector}'
        #
        # Detector stage
        #
        reac_spectrum_at_detector:
          expr: 'baselinewight_switch*reac_spectrum_oscillated'
          label: '{reactor} spectrum at {detector}'
        observable_spectrum_reac:
          expr: 'ibd_xsec_rescaled*reac_spectrum_at_detector'
          label: 'Observable spectrum from {reactor} at {detector}'
        observable_spectrum:
          expr: 'sum:r|observable_spectrum_reac'
          label: 'Observable spectrum at {detector}'
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
        power_livetime_factor:
          expr: 'power_livetime_factor'
          label: '{{Power-livetime factor (~nu/s)|{reactor}.{isotope}-\\>{detector}}}'
        numerator:
          expr: 'numerator'
          label: '{{Power-livetime factor (~MW)|{reactor}.{isotope}-\\>{detector}}}'
        power_livetime_scale:
          expr: 'eff*livetime*thermal_power_scale*thermal_power_nominal*conversion_factor*target_protons'
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
