from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna import constructors as C
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R

class exp(baseexp):
    """\
JUNO+TAO experiment implementation v04d

Presented on 2021.03.22.

Changes since junotao_v04d:
    - Add subsitution lsnl
    - Switch to new input file JUNOInputs2021_03_09.root
    - Switch to the MSW oscprob by Khan et al.
    - Take into account the electron mass when converting the hydrogen_fraction to number of protons, see juno_input_numbers_v2
    - [temprorary] Add an option to set cosθ=0
    - [temprorary] Replace bump with 0.95
    - Remove the xsec switch
    - Temporarily comment TAO until the enu_matrix/enu_subst are correctly dispatched

Derived from:
    - [2021.03.15] junotao_v04e
    - [2021.03.10] junotao_v04d
    - [2021.03.02] junotao_v04c
    - [2021.02.22] junotao_v04b
    - [2021.02] junotao_v04
    - [2021.01] junotao_v03
    - [2020.12] junotao_v02
    - [2020.12] junotao_v01
    - [2020.08] juno_sensitivity_v03
    - [2020.06] juno_sensitivity_v02
    - [2020.04] juno_sensitivity_v01
    - [2019.12] juno_chengyp
    - [2019.12] Daya Bay model from dybOscar and GNA
"""

    nameslib = open('packages/junosens_v2/experiments/junotao_v04b.yaml', 'r').read()
    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')

        # Energy model
        emodel = parser.add_argument_group('emodel', description='Energy model parameters')
        emodel.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres'], default=['lsnl', 'eres'], help='Energy model components')
        emodel.add_argument('--lsnl', choices=['matrix', 'subst'], default='subst', help='LSNL implementation method')

        # Binning
        binning=parser.add_argument_group('binning', description='Binning related options')
        binning.add_argument('--final-emin', type=float, help='Final binning Emin')
        binning.add_argument('--final-emax', type=float, help='Final binning Emax')
        binning.add_argument('--fine-step', type=float, choices=(1.0,), help='Fine integration binning step, keV')
        binning.add_argument('--comparison', action='store_true', help='eneable "comparison" mode')

        # Spectrum
        spectrum=parser.add_argument_group('spectrum', description='Antineutrino spectrum related options')
        parser.add_argument('--free-pars-mode', choices=['log', 'plain'], default=None, help='type of the spectral uncertainty')
        parser.add_argument('--spectra-unc-hm', action='store_true', help='enable HM spectral uncertainties')

        # osc prob
        parser.add_argument('--oscprob', choices=['vacuum', 'matter-approx'], default='matter-approx', help='oscillation probability type')
        parser.add_argument('--msw', choices=['khan-approx', 'juno-yb'], help='MSW oscillation option')
        parser.add_argument('--half-electron-density', action='store_true', help='Use 0.5 for the electron density')

        # Parameters
        correlations = [ 'lsnl', 'subdetectors' ]
        parser.add_argument('--correlation',  nargs='*', default=correlations, choices=correlations, help='Enable correalations')

        # Miscellaneous
        parser.add_argument('--collapse', action='store_true', help='collapsed configuration' )

    def init(self):
        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.build()
        self.parameters()
        self.register()
        self.autodump()

        if self.opts.stats:
            self.print_stats()

    def init_nidx(self):
        # if 'multieres' in self.opts.energy_model:
            # self.subdetectors_names = ['subdet%03i'%i for i in range(5)]
        # else:
        self.subdetectors_names = ()
        self.reactors = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'DYB', 'HZ']

        self.nidx = [
            ('d', 'detector',    ['juno']),
            ['r', 'reactor',     self.reactors],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('rt', 'reactors_tao', ['TS1']),
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('s', 'subdetector', self.subdetectors_names),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] ),
        ]
        if self.opts.collapse:
            self.nidx[1][2] = self.nidx[1][2][6:7]
            self.nidx[2][2] = self.nidx[2][2][:1]
        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        energy_model = self.opts.energy_model
        # if 'eres' in energy_model and 'multieres' in energy_model:
            # raise Exception('Energy model options "eres" and "multieres" are mutually exclusive: use only one of them')

        #
        # Optional formula parts
        #
        keeplsnl = 'lsnl' in energy_model
        keeperes = 'eres' in energy_model
        lsnlsubst = keeplsnl and self.opts.lsnl=='subst'
        lsnlmatrix = keeplsnl and self.opts.lsnl=='matrix'
        print("Energy model:", lsnlsubst and 'LSNL subst,' or lsnlmatrix and 'LSNL matrix,' or 'no LSNL,', keeperes and 'eres' or 'no eres')
        formula_options = dict(
                #
                # Oscillation probability
                #
                oscprob = {
                    'vacuum':        'sum[c]| pmns[c]*oscprob[c,d,r](enu_juno())' ,
                    'matter':        'oscprob_matter[d,r](enu_juno())',
                    'matter-approx': 'oscprob_msw_approx[d,r](enu_juno())',
                    }[self.opts.oscprob],
                #
                # Geo neutrino
                #
                # geonu_spectrum = '+geonu_scale*bracket(geonu_norm_U238*geonu_spectrum_U238(enu_matrix()) + geonu_norm_Th232*geonu_spectrum_Th232(enu_matrix()))'
                                 # if 'geo' in self.opts.bkg else '',
                #
                # Energy model
                #
                lsnlmatrix = 'lsnl|'     if lsnlmatrix else '',
                lsnl_tao   = 'lsnl_tao|' if keeplsnl else '',
                eres = 'eres|' if keeperes else '',
                       # 'concat[s]| rebin| subdetector_fraction[s] * eres[s]|' if 'multieres' in energy_model else
                )

        self.formula = [
                # Some common definitions
                # Physics constants
                'NeutronLifeTime',
                'ElectronMass',
                # Juno related numbers
                'baseline_tao_m',
                'baseline[d,r]',
                'livetime=bracket(daq_years*seconds_in_year)',
                'livetime_tao=bracket(daq_years_tao*seconds_in_year)',
                'conversion_factor',
                'efflivetime = bracket(eff * livetime)',
                'efflivetime_tao = bracket(eff_tao * livetime_tao)',
                # 'geonu_scale = eff * livetime[d] * conversion_factor * target_protons',
                #
                # Integration energy
                #
                'evis_hist2a=evis_hist2a()',
                'evis_hist2b=evis_hist2b()',
                #
                # Energy model
                #
                # JUNO substitution and matrix
                'lsnl_coarse = sum[l]| lsnl_weight[l] * lsnl_component_y[l]()',
                'lsnl_gradient = sum[l]| lsnl_weight[l] * lsnl_component_grad[l]()',
                'lsnl_interpolator| lsnl_x(), lsnl_coarse, evis_edges2a(), evis() ',
                'lsnl_interpolator_grad| lsnl_gradient',
                'lsnl_edges| evis_hist2a',
                # TAO matrix
                'lsnl_coarse_tao = sum[l]| lsnl_weight_tao[l] * lsnl_component_y_tao[l]()',
                'lsnl_interpolator_tao| lsnl_x_tao(), lsnl_coarse_tao, evis_edges2b() ',
                'lsnl_edges_tao| evis_hist2b',
                # Eres
                'eres_matrix| evis_hist2a' if keeperes else '',
                #
                # Neutrino energy
                #
                'ctheta_switch| ctheta_zero(ctheta()), ctheta()',
                'ctheta_mesh_switch| ctheta_mesh_zero(ctheta()), ctheta()',
                'enu_matrix| ee_matrix(evis()), ctheta_switch()',
                'enu_subst| ee_subst(lsnl_evis()), ctheta_switch()',
                'enu_juno=enu_subst' if lsnlsubst else 'enu_juno=enu_matrix',
                # Oscillations
                #
                'pmns[c]',
                #
                # Worst case spectral distortions
                #
                'pmns_wc_no[c]',
                'pmns_wc_io[c]',
                'baseline_wc',
                'Oscprob_wc_full_no = sum[c]| pmns_wc_no[c]*oscprob_wc_no[c](enu_juno())',
                'Oscprob_wc_full_io = sum[c]| pmns_wc_io[c]*oscprob_wc_io[c](enu_juno())',
                'SpectralDistortion = Oscprob_wc_full_io/Oscprob_wc_full_no',
                #
                # Reactor part
                #
                'numerator = thermal_power_scale[r] * thermal_power_correction[r] * thermal_power_nominal[r] * '
                             'fission_fractions_scale[r,i] * fission_fractions_nominal[r,i]() * '
                             'conversion_factor',
                'isotope_weight = energy_per_fission_scale[i] * energy_per_fission[i] * fission_fractions_scale[r,i]',
                'energy_per_fission_avg = sum[i]| isotope_weight * fission_fractions_nominal[r,i]()',
                'power_livetime_factor = numerator / energy_per_fission_avg',
                'anuspec_unc_corrected|enu_juno(), anuspec_unc_scale[i]()*anuspec[i](enu_unc())',
                #
                # SNF
                #
                'energy_per_fission_avg_nominal = sum[i] | energy_per_fission[i] * fission_fractions_nominal[r,i]()',
                'snf_plf_daily = conversion_factor * thermal_power_correction[r] * thermal_power_nominal[r] * fission_fractions_nominal[r,i]() / energy_per_fission_avg_nominal',
                'nominal_spec_per_reac =  sum[i]| snf_plf_daily*anuspec_unc_corrected[i]()',
                'snf_in_reac = snf_norm * snf_correction(enu_juno(), nominal_spec_per_reac)',
                #
                # Backgrounds in JUNO
                #
                'acc_num    = days_in_second * livetime * acc_rate    * acc_rate_norm',
                'fastn_num  = days_in_second * livetime * fastn_rate  * fastn_rate_norm',
                'alphan_num = days_in_second * livetime * alphan_rate * alphan_rate_norm',
                'lihe_num   = days_in_second * livetime * lihe_rate   * lihe_rate_norm',
                'geonu_num  = days_in_second * livetime * geonu_rate  * geonu_rate_norm',
                #
                'geonu_spectrum = frac_Th232 * geonu_Th232_spectrum() + frac_U238 * geonu_U238_spectrum()',
                #
                'accidentals = bracket| acc_num    * rebin_acc[d]   | acc_spectrum()',
                'fastn       = bracket| fastn_num  * rebin_fastn[d] | fastn_spectrum()',
                'alphan      = bracket| alphan_num * rebin_alphan[d]| alphan_spectrum()',
                'lihe        = bracket| lihe_num   * rebin_lihe[d]  | lihe_spectrum()',
                'geonu       = bracket| geonu_num  * rebin_geonu[d] | geonu_spectrum',
                'bkg_juno    = bracket| accidentals + lihe + alphan + fastn + geonu',
                'bkg_shape_variance = bin_width_factor * bkgbin_widths(accidentals) * sumsq_snapshot| sumsq_bkg| lihe_bin2bin*lihe, fastn_bin2bin*fastn, alphan_bin2bin*alphan, geonu_bin2bin*geonu',
                #
                # Backgrounds in TAO
                #
                'accidentals_tao  = bracket| days_in_second * livetime_tao * acc_rate_tao   * acc_rate_norm_tao   * rebin_acc_tao         | acc_spectrum_tao()',
                'lihe_tao         = bracket| days_in_second * livetime_tao * lihe_rate_tao  * lihe_rate_norm_tao  * rebin_lihe_tao        | lihe_spectrum_tao()',
                'fastn_tao        = bracket| days_in_second * livetime_tao * fastn_rate_tao * fastn_rate_norm_tao * rebin_fastn_tao       | fastn_spectrum_tao()',
                'fastn_sigma_tao  = bracket| days_in_second * livetime_tao * fastn_rate_tao * fastn_rate_norm_tao * rebin_fastn_sigma_tao | fastn_shape_unc_rel()*fastn_spectrum_tao()',
                'bkg_tao          = bracket| accidentals_tao + lihe_tao + fastn_tao',
                'bkg_shape_variance_tao = bin_width_factor * bkgbin_tao_widths(accidentals_tao) * sumsq_snapshot_tao| sumsq_bkg_tao| lihe_bin2bin*lihe_tao, fastn_bin2bin_tao*fastn_sigma_tao',
                #
                # IBD part
                #
                '''anuspec_rd_full = reactor_active_norm
                                        *( sum[i]| power_livetime_factor
                                                   *offeq_correction[i,r](enu_juno(), anuspec_unc_corrected[i]())
                                        )
                                        + snf_in_reac''',
                'bump_correction_base| bump_correction_coarse_centers(bump_correction_coarse()), enu_juno()',
                'bump_correction| bump_correction_coarse()',
                'bump_switch| bump_095(bump_correction()), bump_correction()',
                '''cross_section_matrix = bracket(
                                    ibd_xsec_matrix(enu_matrix(), ctheta_mesh_switch())*
                                    jacobian_matrix(enu_matrix(), ee_matrix(), ctheta_switch())*
                                    bump_switch()
                                )''',
                '''cross_section_subst = bracket(
                                    (
                                        ibd_xsec_subst(enu_subst(), ctheta_mesh_switch())*
                                        jacobian_subst(enu_subst(), ee_subst(), ctheta_switch())*
                                        bump_switch()
                                    )
                                    //lsnl_interpolator_grad()
                                )''',
                'cross_section_juno=cross_section_subst' if lsnlsubst else 'cross_section_juno=cross_section_matrix',
                '''ibd=norm_reac*norm_juno*rebin|
                         {eres} {lsnlmatrix}
                           rebin_juno_internal |
                               kinint2_juno(
                                   DistortSpectrum|
                                       sum[r](
                                           (
                                               (baselineweight[r,d]*efflivetime*target_protons*duty_cycle)
                                               * anuspec_rd_full
                                               * {oscprob}
                                           )*cross_section_juno
                                       ),
                                       SpectralDistortion
                               )
                '''.format(**formula_options),
                #
                # Total observation
                #
                'observation=ibd + bkg_juno',
                'variance_juno = staterr2(observation) + bkg_shape_variance',
                # #
                # # TAO part
                # #
                # 'eres_matrix_tao| evis_hist2b',
                # 'anuspec_rd_full_tao = select1[r,"TS1"]| anuspec_rd_full',
                # '''ibd_tao = norm_reac*norm_tao*rebin_tao| eres_tao|
                               # {lsnl_tao}
                                 # eleak_tao|
                                   # rebin_tao_internal|
                                       # kinint2_tao|
                                         # DistortSpectrumTAO(
                                           # sum[rt](
                                             # bracket(
                                                # (baselineweight_tao[rt]*efflivetime_tao*target_protons_tao*duty_cycle_tao)
                                               # * anuspec_rd_full_tao * cross_section_matrix)),
                                           # SpectralDistortion
                                         # )
                             # '''.format(**formula_options),
                # #
                # # TAO total observation
                # #
                # 'observation_tao=ibd_tao + bkg_tao',
                # 'variance_tao = staterr2_tao(observation_tao) + bkg_shape_variance_tao',
                #
                # Comparison cases
                #
                'cmp_accidentals = acc_num    * acc_spectrum()',
                'cmp_fastn       = fastn_num  * fastn_spectrum()',
                'cmp_alphan      = alphan_num * alphan_spectrum()',
                'cmp_lihe        = lihe_num   * lihe_spectrum()',
                'cmp_geonu       = geonu_num  * geonu_spectrum',
                'cmp_bkg_juno    = cmp_rebin_bkg[d]| cmp_accidentals + cmp_lihe + cmp_alphan + cmp_fastn + cmp_geonu',
                'cmp_juno_case12 = cmp_rebin_12| norm_reac*norm_juno*rebin_juno_internal()',
                'cmp_juno_case3  = cmp_rebin_3| norm_reac*norm_juno*eres()',
                'cmp_juno_case4  = cmp_rebin_4(cmp_juno_case3)+cmp_bkg_juno',
                ]

    def parameters(self):
        ns = self.namespace
        dmxx = 'pmns.DeltaMSq13'
        for par in [dmxx, 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']:
            ns[par].setFree()

    def init_configuration(self):
        step = 0.005
        if self.opts.fine_step:
            step = self.opts.fine_step*1.e-3
        step_tao = 0.01
        edges_internal = np.arange(0.0, 12.001, step)
        edges_internal_tao = np.arange(0.0, 12.001, step_tao)
        juno_edges = np.concatenate( (
                                    [0.8],
                                    np.arange(1, 6.0, 0.02),
                                    np.arange(6, 7.0, 0.1),
                                    [7.0, 7.5, 12.0]
                                )
                            )

        if self.opts.final_emin is not None:
            print('Truncate final binning E>={}'.format(self.opts.final_emin))
            juno_edges = juno_edges[juno_edges>=self.opts.final_emin]
        if self.opts.final_emax is not None:
            print('Truncate final binning E<={}'.format(self.opts.final_emax))
            juno_edges = juno_edges[juno_edges<=self.opts.final_emax]
        if self.opts.final_emin is not None or self.opts.final_emax is not None:
            print('Final binning:', juno_edges)

        tao_edges, enu_edges = self.init_tao_binning()
        if self.opts.comparison:
            print('Running in comparison mode! Enu bins modified')
            a, b, c = enu_edges[enu_edges<1.99], enu_edges[(enu_edges>=1.99)*(enu_edges<8.01)], enu_edges[enu_edges>=8.01]
            enu_edges = np.concatenate([a, np.arange(2.0, 8.01, 0.010), c], dtype='d')
        anue_spec_unc_edges = enu_edges[(enu_edges>0.0)*(enu_edges<=8.0)]
        anue_spec_unc_edges = np.arange(anue_spec_unc_edges[0], anue_spec_unc_edges[-1]+1.e-6, 0.01)
        anue_spec_unc_edges = np.concatenate([anue_spec_unc_edges, enu_edges[enu_edges>8.0]])

        from physlib import pc
        energy_offset = pc.DeltaNP - pc.ElectronMass
        enu_threshold=1.8
        EvisToEnu = lambda Evis: Evis + energy_offset

        tao_edges = np.concatenate([juno_edges[juno_edges<=6.0], tao_edges[(tao_edges>6.0)*(tao_edges<8.0)], [8.0, 9.0, 12.0]])
        tao_edges[0] = 0.9
        enu_edges1 = EvisToEnu(tao_edges[tao_edges>=8.0])
        enu_edges = np.concatenate([enu_edges[enu_edges<8.0], enu_edges1[:-1]])


        if self.opts.verbose:
            print('JUNO binning:', format_binning(juno_edges, 2))
            print('TAO binning:', format_binning(tao_edges, 2))
            print('Enu binning:',  format_binning(enu_edges, 4))
            print('Enu spec. unc. binning',  format_binning(anue_spec_unc_edges, 4))

        common_input_location='data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input'
        common_input_root_file='deprecated/JUNOInputs2021_03_09.root'
        self.cfg = NestedDict(
                #
                # Numbers
                #
                numbers0 = dict(
                    bundle = dict(name='parameters', version='v05'),
                    state='fixed',
                    labels=dict(
                        seconds_in_year   = 'Number of seconds in year',
                        days_in_second    = 'Number of days in a second',
                        conversion_factor = 'Conversion factor: W[GW]/[MeV] N[fissions]->N[fissions]/T[s]',
                        ),
                    pars =  dict(
                            seconds_in_year   = 365.25*24.0*60.0*60.0,
                            days_in_second    = 1.0/(24.0*60.0*60.0),
                            conversion_factor = R.NeutrinoUnits.reactorPowerConversion, #taken from transformations/neutrino/ReactorNorm.cc
                            ),
                    ),
                numbers = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/deprecated/files/juno_input_numbers_v2.py',
                    skip = ('percent',),
                    state= 'fixed'
                    ),
                bkg_shape_unc = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/bkg_bin2bin.yaml',
                    hooks = dict(bin_width_factor=lambda pars: (1.0/pars['bin_width'], '1/`bin width`')),
                    state= 'fixed'
                    ),
                tao_numbers = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/tao_input_numbers.py',
                    skip = ('percent',),
                    state= 'fixed'
                    ),
                baselines_tao = dict(
                        bundle = dict(name='reactor_baselines', version='v03', major=('rt', ''), names=lambda s: s+'_tao'),
                        reactors  = f'{common_input_location}/files/reactor_baselines_tao.yaml',
                        reactors_key = 'reactor_baseline',
                        detectors = {None: 0.0},
                        unit    = 'm',
                        label   = "Baseline between TAO and {reactor}, km",
                        label_w = "1/(4πL²) for TAO and {reactor}, cm⁻²"
                        ),
                # Detector and reactor
                norm_reac = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "norm_reac",
                        label = 'Reactor power efficiency normalization',
                        pars = uncertain(1.0, 2, 'percent')
                        ),
                norm_juno = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'norm_juno',
                        label = 'Detection efficiency normalization JUNO',
                        pars = uncertain(1.0, 1, 'percent')
                        ),
                norm_tao = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'norm_tao',
                        label = 'Detection efficiency normalization TAO',
                        pars = uncertain(1.0, 1, 'percent')
                        ),
                baselines = dict(
                        bundle = dict(name='reactor_baselines', version='v02', major = 'rd'),
                        reactors  = f'{common_input_location}/files/reactor_baselines.yaml',
                        reactors_key = 'reactor_baseline',
                        detectors = dict(juno=0.0),
                        unit = 'km'
                        ),
                # Reactor
                energy_per_fission =  dict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = f'{common_input_location}/files/energy_per_fission.yaml',
                        separate_uncertainty = '{}_scale'
                        ),
                thermal_power =  dict(
                        bundle = dict(name="parameters", version = "v06", names=dict(thermal_power='thermal_power_nominal')),
                        pars = f'{common_input_location}/files/thermal_power.yaml',
                        separate_uncertainty = '{}_scale'
                        ),
                thermal_power_correction =  dict(
                        bundle = dict(name="parameters", version = "v06", names=dict(thermal_power='thermal_power_nominal')),
                        state='fixed',
                        labels=dict(
                            thermal_power_correction = 'Correction scale for the thermal power',
                            ),
                        pars =  dict(
                            thermal_power_correction = dict(
                                TS1=1.0, TS2=1.0,
                                YJ1=1.0, YJ2=1.0, YJ3=1.0,
                                YJ4=1.0, YJ5=1.0, YJ6=1.0,
                                DYB=1.0,
                                HZ=0.5
                                )
                            ),
                        ),
                # Backgrounds
                bkg_rate = dict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = f'{common_input_location}/files/bkg_rates_v3.yaml',
                        separate_uncertainty = '{}_norm'
                    ),
                bkg_rate_tao = dict(
                        bundle = dict(name="parameters", version = "v06", names='_tao'),
                        pars = f'{common_input_location}/files/bkg_rates_tao.yaml',
                        separate_uncertainty = '{}_norm'
                    ),
                #
                # Transformations
                #
                # General
                kinint2 = dict(
                    bundle    = dict(name='integral_2d1d', version='v05'),
                    variables = ('evis', 'ctheta'),
                    xedgescfg  = [
                                   (0.0, step,   0),
                                   (0.8, step/4, 3),
                                   (1.1, step,   3),
                                   (12, None, None),
                                ],
                    yorder    = 5,
                    instances = {
                        'kinint2_juno': 'JUNO integral',
                        'kinint2_tao': {'label': 'TAO integral', 'noindex': True},
                        }
                    ),
                rebin_internal_juno = dict(
                    bundle = dict(name='rebin', version='v05', major='', names={'rebin_hist': 'evis_hist2a', 'rebin_points': 'evis_edges2a'}),
                    rounding = 6,
                    edges = edges_internal,
                    instances={
                        'rebin_juno_internal': 'JUNO internal binning',
                        }
                    ),
                rebin_internal_tao = dict(
                    bundle = dict(name='rebin', version='v05', major='', names={'rebin_hist': 'evis_hist2b', 'rebin_points': 'evis_edges2b'}),
                    rounding = 6,
                    edges = edges_internal_tao,
                    instances={
                        'rebin_tao_internal':  'TAO internal binning'
                        }
                    ),
                rebin = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = juno_edges,
                    instances={ 'rebin': 'JUNO final' }
                    ),
                rebin_bkg = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = juno_edges,
                    instances={
                        'rebin_acc': 'Accidentals {autoindex}',
                        'rebin_lihe': '9Li/8He {autoindex}',
                        'rebin_fastn': 'Fast neutrons {autoindex}',
                        'rebin_alphan': 'C(alpha,n)O {autoindex}',
                        'rebin_geonu': 'Geo-nu {autoindex}',
                        }
                    ),
                rebin_bkg_tao = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = tao_edges,
                    instances={
                        'rebin_acc_tao': 'Acc. TAO',
                        'rebin_lihe_tao': '9Li/8He TAO',
                        'rebin_fastn_tao': 'Fast neutrons TAO',
                        'rebin_fastn_sigma_tao': 'Fast neutrons TAO',
                        }
                    ),
                rebin_tao = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = tao_edges,
                    instances={ 'rebin_tao': 'TAO final' }
                    ),
                rebin_comparison = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = np.arange(0.8, 12.0+1.e-6, 0.02),
                    instances={
                        'cmp_rebin_bkg': 'Rebin comp. bkg.',
                        'cmp_rebin_4': 'Rebin comp. 4',
                        }
                    ),
                rebin_comparison5 = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = np.arange(0.8, 12.0+1.e-6, 0.005),
                    instances={
                        'cmp_rebin_3': 'Rebin comp. 3',
                        'cmp_rebin_12': 'Rebin comp. 1&2',
                        }
                    ),
                bkg_shape_uncertainty = dict(
                    bundle = dict(name='trans_sumsq', version='v01', major=''),
                    ninputs = 4,
                    instances={'sumsq_bkg': 'Bkg shape variance {autoindex}'}
                    ),
                bkg_shape_uncertainty_tao = dict(
                    bundle = dict(name='trans_sumsq', version='v01', major=''),
                    ninputs = 2,
                    instances={'sumsq_bkg_tao': 'Bkg shape variance TAO'}
                    ),
                bkg_edges = dict(
                    bundle = dict(name='trans_histedges', version='v01', major=''),
                    types = ('widths', ),
                    instances={
                        'bkgbin': 'Bkg bins {autoindex}'
                        }
                    ),
                bkg_edges_tao = dict(
                    bundle = dict(name='trans_histedges', version='v01', major=''),
                    types = ('widths', ),
                    instances={
                        'bkgbin_tao': 'Bkg bins TAO'
                        }
                    ),
                variance = dict(
                    bundle = dict(name='trans_snapshot', version='v01', major=''),
                    instances={'sumsq_snapshot': 'Bkg shape variance snapshot, not corrected|{autoindex}',
                               'staterr2': 'Stat. errors (snapshot)'}
                    ),
                variance_tao = dict(
                    bundle = dict(name='trans_snapshot', version='v01', major=''),
                    instances={'sumsq_snapshot_tao': 'Bkg shape variance snapshot, not corrected|TAO',
                               'staterr2_tao': 'Stat. errors (snapshot) TAO'}
                    ),
                numbers_tao_var = dict(
                    bundle = dict(name='parameters', version='v05'),
                    state='fixed',
                    labels=dict(
                        fastn_bin2bin_tao   = 'Fast neutron bin2bin switch',
                        ),
                    pars =  dict(
                            fastn_bin2bin_tao = 1.0,
                            ),
                    ),
                # TAO detector
                eleak_tao = dict(
                    bundle     = dict(name='detector_energy_leakage_root', version='v01', names=lambda s: s+'_tao'),
                    filename   = f'{common_input_location}/{common_input_root_file}',
                    matrixname = 'TAO_response_matrix_25',
                    ),
                eres_tao = dict(
                        bundle = dict(name='detector_eres_inputsigma', version='v01', major='', names=lambda s: s+'_tao'),
                        filename = f'{common_input_location}/files/tao_eres_sigma_fine_1200.dat',
                        expose_matrix = False
                        ),
                #
                # IBD
                #
                ibd_constants1 = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/pdg2014.yaml',
                    ),
                ibd_constants2 = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/pdg2020.yaml',
                    skip = ('NeutronLifeTime', )
                    ),
                ibd_xsec_vogel_subst = dict(
                    bundle = dict(name='xsec_ibd', version='v04', names='_subst'),
                    order = 1,
                    verbose = 0,
                    constants = {
                        'PhaseFactor': 1.71465,
                        'g':           1.2701,
                        'f':           1.0,
                        'f2':          3.706,
                        }
                    ),
                ibd_xsec_vogel_matrix = dict(
                    bundle = dict(name='xsec_ibd', version='v04', names='_matrix'),
                    order = 1,
                    verbose = 0,
                    constants = {
                        'PhaseFactor': 1.71465,
                        'g':           1.2701,
                        'f':           1.0,
                        'f2':          3.706,
                        }
                    ),
                #
                # cosθ overriding
                #
                costheta_filllike = dict(
                        bundle = dict(name='filllike', version='v01'),
                        instances = {
                            'ctheta_zero': 'cosθ=0',
                            'ctheta_mesh_zero': 'cosθ=0 (mesh)',
                            },
                        value = 0.0
                        ),
                costheta_switch = dict(
                        bundle = dict(name='switch', version='v01', names={'condition': 'ctheta_switchvar'}),
                        instances = {
                            'ctheta_switch': 'cosθ=0 or cosθ',
                            'ctheta_mesh_switch': 'cosθ=0 or cosθ',
                            },
                        varlabel = 'cosθ(=/≠)0 switch',
                        ninputs = 2,
                        default = 1
                        ),
                #
                # Oscillations
                #
                oscpars = dict(
                        bundle = dict(name='oscpars_ee', version='v01'),
                        fixed = False,
                        parameters = dict(
                            # DeltaMSq23    = 2.453e-03,
                            DeltaMSq13    = 2.5283e-3,
                            DeltaMSq12    = 7.53e-05,
                            SinSqDouble13 = (0.08529904, 0.00267792),
                            SinSqDouble12 = 0.851004,
                            )
                        ),
                oscprob = dict(
                        bundle = dict(name='oscprob_ee', version='v01', major='rdc', inactive=self.opts.oscprob!='vacuum'),
                        ),
                oscprob_msw_approx = dict(
                        bundle = dict(name='oscprob_approx', version='v02', major='rd', inactive=self.opts.oscprob!='matter-approx'),
                        formula = self.opts.msw or 'khan-approx',
                        density=2.45,
                        electron_density=self.opts.half_electron_density and 0.5 or None
                        ),
                oscprob_matter = dict(
                    bundle = dict(name='oscprob_matter', version='v02', major='rd', inactive=self.opts.oscprob=='vacuum',
                                  names=dict(oscprob='oscprob_matter')),
                    density = 2.45, # g/cm3
                    ),
                #
                # Worst case spectral distortion: oscillation probability
                #
                numbers_wc = dict(
                    bundle = dict(name='parameters', version='v05'),
                    state='fixed',
                    labels=dict(
                        baseline_wc = 'Baseline for worst case distortion, km',
                        ),
                    pars =  dict(
                            baseline_wc = 52.5301918767,
                            ),
                    ),
                oscpars_wc_no = dict(
                        bundle = dict(name='oscpars_ee', version='v01', names={'pmns': 'pmns_wc_no'}),
                        fixed = True,
                        parameters = dict(
                            # DeltaMSq23    = 2.453e-03,
                            DeltaMSq13    = 2.5283e-3,
                            DeltaMSq12    = 7.53e-05,
                            SinSqDouble13 = (0.08529904, 0.00267792),
                            SinSqDouble12 = 0.851004,
                            )
                        ),
                oscpars_wc_io = dict(
                        bundle = dict(name='oscpars_ee', version='v01', names={'pmns': 'pmns_wc_io'}),
                        fixed = True,
                        parameters = dict(
                            # DeltaMSq23    = 0.00256153884886  ,
                            DeltaMSq13    = 0.00256153884886+7.52800519162e-05,
                            DeltaMSq12    = 7.52800519162e-05 ,
                            SinSqDouble13 = 0.0816211422423   ,
                            SinSqDouble12 = 0.851602205229    ,
                            )
                        ),
                oscprob_wc_no = dict(
                        bundle = dict(name='oscprob_ee', version='v01', major=('', '', 'c'),
                                      names={'baseline': 'baseline_wc', 'pmns': 'pmns_wc_no', 'oscprob': 'oscprob_wc_no'}),
                        dmnames = ['DeltaMSq12', 'DeltaMSq13', 'DeltaMSq23NO'],
                        labelfmt = 'OP {component}|worst case NO'
                        ),
                oscprob_wc_io = dict(
                        bundle = dict(name='oscprob_ee', version='v01', major=('', '', 'c'),
                                      names={'baseline': 'baseline_wc', 'pmns': 'pmns_wc_io', 'oscprob': 'oscprob_wc_io'}),
                        dmnames = ['DeltaMSq12', 'DeltaMSq13', 'DeltaMSq23IO'],
                        labelfmt = 'OP {component}|worst case IO'
                        ),
                condproduct = dict(
                        bundle = dict(name='conditional_product', version='v01', major=(),
                                      names={'condition': 'distortion_wc_on'}),
                        instances = {
                            'DistortSpectrum': '{{Optionally distorted spectrum | worst case}}',
                            },
                        condlabel = 'Worst case distortion switch',
                        default   = 0
                        ),
                condproduct_tao = dict(
                        bundle = dict(name='conditional_product', version='v01', major=(),
                                      names={'condition': 'distortion_wc_on'}),
                        instances = {
                            'DistortSpectrumTAO': '{{Optionally distorted spectrum [TAO] | worst case}}',
                            },
                        condlabel = 'Worst case distortion switch',
                        default   = 0
                        ),
                #
                # Reactor antineutrino spectrum
                #
                anuspec_hm = dict(
                        bundle = dict(name='reactor_anu_spectra', version='v06'),
                        name = 'anuspec',
                        filename = f'{common_input_location}/{common_input_root_file}',
                        objectnamefmt = 'HuberMuellerFlux_{isotope}',
                        spectral_parameters='fixed',
                        varmode=self.opts.free_pars_mode or 'log',
                        varname='anue_weight_{index:04d}',
                        ns_name='spectral_weights',
                        edges=enu_edges
                        ),
                anuspec_hm_unc = dict(
                        bundle = dict(name='reactor_anu_spectra_unc', version='v01'),
                        name = 'anuspec_unc',
                        ename = 'enu_unc',
                        filename = ['data/data_juno/anue_spectra/hm_interpolation/10_keV_bin/Huber_{type}rrelated_unc_extrap_{isotope}_13.0_0.01_MeV.txt',
                                    'data/data_juno/anue_spectra/hm_interpolation/10_keV_bin/Mueller_{type}rrelated_unc_extrap_{isotope}_13.0_0.01_MeV.txt'],
                        varname_uncor='anue_unc_uncor_{index:04d}',
                        varname_cor='anue_unc_cor',
                        ns_name='spectral_unc',
                        edges=anue_spec_unc_edges,
                        fixed=not self.opts.spectra_unc_hm,
                        debug=False
                        ),
                offeq_correction = dict(
                        bundle = dict(name='reactor_offeq_spectra', version='v05', major=''),
                        offeq_data = f'{common_input_location}/{common_input_root_file}',
                        objectnamefmt = 'NonEq_FluxRatio',
                        relsigma = 0.3,
                        ),
                snf_correction = dict(
                        bundle = dict(name='reactor_snf_spectra', version='v05', major='r'),
                        snf_average_spectra = f'{common_input_location}/{common_input_root_file}',
                        objectname = 'SNF_FluxRatio'
                        ),
                #
                # Bump correction
                #
                bump_correction = dict(
                        bundle = dict(name='interpolation_1d', version='v01'),
                        name='bump_correction',
                        kind='expo',
                        strategy='nearestedge',
                        label='Bump correction idx',
                        labelfmt='Bump correction',
                        ),
                bump_correction_coarse_centers = dict(
                        bundle = dict(name='trans_histedges', version='v01'),
                        types = ('centers', ),
                        instances={
                            'bump_correction_coarse': 'Bump correction centers'
                            }
                        ),
                bump_correction_in = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/{common_input_root_file}',
                    formats = ['DYBFluxBump_ratio'],
                    names   = ['bump_correction_coarse'],
                    labels  = ['HM bump correction (coarse)'],
                    normalize = False,
                    ),
                #
                # Bump overriding
                #
                bump_filllike = dict(
                        bundle = dict(name='filllike', version='v01'),
                        instances = {
                            'bump_095': 'n=0.95',
                            },
                        value = 0.95
                        ),
                bump_switch = dict(
                        bundle = dict(name='switch', version='v01', names={'condition': 'bump_switchvar'}),
                        instances = {
                            'bump_switch': 'n=0.95 or bump',
                            },
                        varlabel = 'bump switch (n=0.95 or bump)',
                        ninputs = 2,
                        default = 0
                        ),
                #
                # Reactor options
                #
                fission_fractions = dict(
                        bundle = dict(name="parameters_yaml_v01", major = 'i'),
                        parameter = 'fission_fractions_nominal',
                        separate_uncertainty = "fission_fractions_scale",
                        label = 'Fission fraction of {isotope} in reactor {reactor}',
                        objectize=True,
                        data = f'{common_input_location}/files/fission_fractions.yaml'
                        ),
                #
                # Detector Effects
                #
                lsnl = dict(
                        bundle = dict(name='energy_nonlinearity_db_root_subst', version='v01', major='l',),
                        names      = {
                            'nominal': 'positronScintNL',
                            'pull0': 'positronScintNLpull0',
                            'pull1': 'positronScintNLpull1',
                            'pull2': 'positronScintNLpull2',
                            'pull3': 'positronScintNLpull3',
                            },
                        filename   = f'{common_input_location}/{common_input_root_file}',
                        extrapolation_strategy = 'extrapolate',
                        nonlin_range = (0.95, 12.),
                        expose_matrix = False
                        ),
                lsnl_tao = dict(
                        bundle = dict(name='energy_nonlinearity_db_root', version='v04', major='l', names=lambda s: s+'_tao'),
                        names  = dict( [
                            ('nominal', 'positronScintNL'),
                            ('pull0', 'positronScintNLpull0'),
                            ('pull1', 'positronScintNLpull1'),
                            ('pull2', 'positronScintNLpull2'),
                            ('pull3', 'positronScintNLpull3'),
                            ]),
                        filename   = f'{common_input_location}/{common_input_root_file}',
                        extrapolation_strategy = 'extrapolate',
                        nonlin_range = (0.95, 12.),
                        expose_matrix = False
                        ),
                shape_uncertainty = dict(
                        unc = uncertain(1.0, 1.0, 'percent'),
                        nbins = 200 # number of bins, the uncertainty is defined to
                        ),
                snf_norm = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'snf_norm',
                        label='SNF norm',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                reactor_active_norm = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'reactor_active_norm',
                        label='Reactor nu (active norm)',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                #
                # Backgrounds
                #
                bkg_spectra = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/{common_input_root_file}',
                    formats = ['AccBkgHistogramAD',           'Li9BkgHistogramAD',       'FnBkgHistogramAD',       'AlphaNBkgHistogramAD',   'GeoNuTh232'                 , 'GeoNuU238'],
                    names   = ['acc_spectrum',                'lihe_spectrum',           'fastn_spectrum',         'alphan_spectrum',        'geonu_Th232_spectrum'       , 'geonu_U238_spectrum'],
                    labels  = ['Acc. JUNO|(norm spectrum)', '9Li/8He JUNO|(norm spectrum)', 'Fast n JUNO|(norm spectrum)', 'AlphaN JUNO|(norm spectrum)', 'GeoNu Th232 JUNO|(norm spectrum)', 'GeoNu U238 JUNO|(norm spectrum)'],
                    normalize = True,
                    ),
                #
                # Backgrounds
                #
                bkg_spectra_tao = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/inputs_ihep/TAOoutfromBackgroundPSD_6year.root',
                    formats = ['histBkgShape_Acc', 'histBkgShape_Li9',  'histBkgShape_Fn'],
                    names   = ['acc_spectrum_tao', 'lihe_spectrum_tao', 'fastn_spectrum_tao'],
                    labels  = ['Accidentals TAO|(norm spectrum)', '9Li/8He TAO|(norm spectrum)', 'Fast n TAO|(norm spectrum)'],
                    normalize = True,
                    ),
                bkg_spectra_unc_tao = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/inputs_ihep/TAOoutfromBackgroundPSD_6year.root',
                    formats = ['FnErr'],
                    names   = ['fastn_shape_unc_rel'],
                    labels  = [ 'Fast n shape unc 6 years|(relative)'],
                    normalize = False,
                    ),
                geonu_fractions=dict(
                        bundle = dict(name='var_fractions_v02'),
                        names = [ 'Th232', 'U238' ],
                        format = 'frac_{component}',
                        fractions = uncertaindict(
                            Th232 = ( 0.23, 'fixed' )
                            ),
                        ),
                )

        if 'eres' in self.opts.energy_model:
            self.cfg.eres = dict(
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
        # elif 'multieres' in self.opts.energy_model:
            # self.cfg.subdetector_fraction = dict(
                    # bundle = dict(name="parameters", version = "v03"),
                    # parameter = "subdetector_fraction",
                    # label = 'Subdetector fraction weight for {subdetector}',
                    # pars = uncertaindict(
                        # [(subdet_name, (0.2, 0.04, 'relative')) for subdet_name in self.subdetectors_names],
                        # ),
                    # covariance = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_cov.txt'
                    # )
            # self.cfg.multieres = dict(
                    # bundle = dict(name='detector_multieres_stats', version='v01', major='s'),
                    # # pars: sigma_e/e = sqrt(b^2/E),
                    # parameter = 'eres',
                    # relsigma = self.opts.eres_b_relsigma,
                    # nph = 'data/data_juno/energy_resolution/2019_subdetector_eres_n200_proper/subdetector5_nph.txt',
                    # rescale_nph = self.opts.eres_npe,
                    # expose_matrix = False
                    # )

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
        self.expression.guessname(self.nameslib, save=True)

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

            if self.opts.verbose>3:
                print('Inputs:')
                print(self.context.inputs.__str__(nested=True, width=width))
                print()

        if self.opts.verbose or self.opts.stats:
            print('Parameters:')
            self.stats = dict()
            correlations = self.opts.verbose>2 and 'full' or 'short'
            self.namespace.printparameters(labels=55, stats=self.stats, correlations=correlations)

    def update_variance(self):
        snapshots = [s for provider in (self.context.providers[name] for name in ['sumsq_snapshot', 'sumsq_snapshot_tao']) for s in provider.bundle.objects]
        outputs = self.context.outputs

        print('Compute and fix JUNO and TAO variance (stat, bkg)')
        for snapshot in snapshots:
            snapshot.nextSample()
        outputs.variance_juno.juno.touch()
        outputs.variance_tao.touch()

    def register(self):
        from gna.env import env
        futurens = env.future.child(('spectra', self.namespace.name))

        outputs = self.context.outputs

        futurens[('variance', 'juno', 'stat')]     = outputs.staterr2.juno
        futurens[('variance', 'juno', 'bkgshape')] = outputs.bkg_shape_variance.juno
        futurens[('variance', 'juno', 'full')]     = outputs.variance_juno.juno
        # futurens[('variance', 'tao', 'stat')]      = outputs.staterr2_tao
        # futurens[('variance', 'tao', 'bkgshape')]  = outputs.bkg_shape_variance_tao
        # futurens[('variance', 'tao', 'full')]      = outputs.variance_tao
        # Force calculation of the stat errors

        env.future[('hooks', self.namespace.name, 'variance')]=lambda: self.update_variance()

        # futurens[('tao', 'initial')]   = outputs.kinint2_tao
        # try:
            # futurens[('tao', 'edep')]  = outputs.eleak_tao
            # futurens[('tao', 'fine')]  = outputs.eleak_tao
        # except: pass
        # try:
            # futurens[('tao', 'equench')] = outputs.lsnl_tao
            # futurens[('tao', 'fine')]    = outputs.lsnl_tao
        # except: pass
        # try:
            # futurens[('tao', 'evis')]  = outputs.eres_tao
            # futurens[('tao', 'fine')]  = outputs.eres_tao
        # except: pass
        # futurens[('tao', 'rebin')]     = outputs.rebin_tao
        # futurens[('tao', 'ibd')]       = outputs.ibd_tao
        # futurens[('tao', 'final')]     = outputs.observation_tao

        futurens[('juno', 'initial')]  = outputs.kinint2_juno.juno
        try:
            futurens[('juno', 'equench')] = outputs.lsnl.juno
            futurens[('juno', 'fine')] = outputs.lsnl.juno
        except: pass
        try:
            futurens[('juno', 'evis')] = outputs.eres.juno
            futurens[('juno', 'fine')] = outputs.eres.juno
        except: pass
        futurens[('juno', 'rebin')]    = outputs.rebin.juno
        futurens[('juno', 'ibd')]      = outputs.ibd.juno
        futurens[('juno', 'final')]    = outputs.observation.juno

        k0 = ('extra',)
        for k, v in self.context.outputs.items(nested=True):
            futurens[k0+k]=v

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.rebin.juno
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    def init_tao_binning(self):
        fname = 'data/local/tao-binning-20.dat'
        evis, enu, _ = np.loadtxt(fname, unpack=True)
        enu=enu[enu!=0.0]
        return evis, enu

import itertools as it
def format_binning(edges, flen):
    edges = np.asanyarray(edges)
    widths = np.round(edges[1:]-edges[:-1], flen+1)*1000.0
    summary = []
    skip=False
    for e, wp, w, wn1, wn2 in it.zip_longest(edges, [-1]+widths.tolist(), widths, widths[1:], widths[2:]):
        if not skip and w==wp: continue
        summary.append(f'{e:.{flen}f}')
        if not w:
            break

        skip=(wn1!=wn2)
        append=', '
        if wn2 and w==wn2:
            append=f', {w:+.0f}…'+append
        summary[-1]+=append

    return '{} bins; '.format(edges.size-1)+''.join(summary)
