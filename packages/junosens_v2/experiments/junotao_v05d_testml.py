from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict, StripNestedDict
from gna import constructors as C
from gna.expression.index import NIndex
import numpy as np
from load import ROOT as R

class exp(baseexp):
    """\
JUNO+TAO experiment implementation v05d_testml

Used to study an impact of ML based energy resolution

Based on version v05d. Updates:
    + JUNO energy resolution
        - Add a switch to choose from default and alternative energy resolution

Derived from:
    - [2021.10.06] junotao_v05d_testml
    - [2021.07.20] junotao_v05c
    - [2021.07.06] junotao_v05b
    - [2021.06.04] junotao_v05a
    - Combined:
        + [2021.05.26] merged in junotao_v04l
        + [2021.05.04] junotao_v04j
    - [2021.04.13 and before] see junotao_v05b for details
"""

    nameslib = open('packages/junosens_v2/experiments/junotao_v05c.yaml', 'r').read()
    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')

        # Consistency check with previous version:
        consistency = parser.add_argument_group('consistency', description='Parameters to check consistency with previous version')
        consistency.add_argument('--consistency-eres', action='store_true', help='Energy resolution ABC from v05b, no correction')
        consistency.add_argument('--enable-hz', '--consistency-hz',   action='store_true', help='Enable HZ reactors')
        consistency.add_argument('--disable-lbl', '--consistency-lbl',  action='store_true', help='Disable LBD and ELBL reactors')

        # Energy model
        emodel = parser.add_argument_group('emodel', description='Energy model parameters')
        emodel.add_argument('--energy-model', nargs='*', choices=['lsnl', 'eres'], default=['lsnl', 'eres'], help='Energy model components')
        # emodel.add_argument('--lsnl', choices=['matrix', 'subst'], default='subst', help='LSNL implementation method')
        emodel.add_argument('--override-eres', nargs=3, help='override energy resolution from pickle file, saved as {reco: [data0, data1, ...]}', metavar=('filename', 'reco', 'num'))

        # Binning
        binning=parser.add_argument_group('binning', description='Binning related options')
        binning.add_argument('--final-emin', type=float, help='Final binning Emin')
        binning.add_argument('--final-emax', type=float, help='Final binning Emax')
        binning.add_argument('--fine-step', type=float, choices=(1.0,), help='Fine integration binning step, keV')
        binning.add_argument('--comparison', action='store_true', help='eneable "comparison" mode')
        binnings = ('var-20', 'var-30', 'var-40', 'const-20', 'const-30', 'const-40')
        binning.add_argument('--binning-tao', default='var-20', choices=binnings, help='base tao binning')

        # Spectrum
        spectrum=parser.add_argument_group('spectrum', description='Antineutrino spectrum related options')
        parser.add_argument('--free-pars-mode', choices=['log', 'plain'], default=None, help='type of the spectral uncertainty')
        parser.add_argument('--spectra-unc-hm', action='store_true', help='enable HM spectral uncertainties')

        # osc prob
        parser.add_argument('--oscprob', choices=['vacuum', 'matter-approx'], default='matter-approx', help='oscillation probability type')
        # parser.add_argument('--half-electron-density', action='store_true', help='Use 0.5 for the electron density')

        # Parameters
        parameters = parser.add_argument_group('parameters', description='Parameter settings')
        parameters.add_argument('--worst-case-fit-pars', action='store_true', help='fix/release parameters for the worst case scenario fit')
        parameters.add_argument('--combination', action='store_true', help='enable combination mode (fix bin2bin nuisance)')

        # Miscellaneous
        parser.add_argument('--collapse', action='store_true', help='collapsed configuration' )

    def init(self):
        self.init_reactors()
        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.build()
        self.parameters()
        self.register()
        self.autodump()

        if self.opts.stats:
            self.print_stats()

    def init_reactors(self):
        if self.opts.enable_hz:
            self.reactors_mbl = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'DYB', 'HZ']
        else:
            self.reactors_mbl = ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'DYB']

        self.reactors_lbl = [
                'FANGCHENGGANG-1', 'FANGCHENGGANG-2',
                'CHANGJIANG-1', 'CHANGJIANG-2',
                'FUQING-1', 'FUQING-2', 'FUQING-3', 'FUQING-4',
                'NINGDE-1', 'NINGDE-2', 'NINGDE-3', 'NINGDE-4',
                'SANMEN-1', 'SANMEN-2',
                'FANGJIASHAN-1', 'FANGJIASHAN-2',
                'QIANSHAN2-1', 'QIANSHAN2-2', 'QIANSHAN2-3', 'QIANSHAN2-4', 'QIANSHAN3-1', 'QIANSHAN3-2',
                'QINSHAN-1',
                'TIANWAN-1', 'TIANWAN-2', 'TIANWAN-3', 'TIANWAN-4',
                'HAIYANG-1'
                ]
        if self.opts.disable_lbl:
            self.reactors_lbl = []

        self.isotopes = ['U235', 'U238', 'Pu239', 'Pu241']

        if self.opts.collapse:
            tsidx = self.reactors_mbl.index('TS1')
            self.reactors_mbl = [self.reactors_mbl[tsidx]]

            self.reactors_lbl=[]

            self.isotopes = [self.isotopes[0]]

        self.reactors = self.reactors_mbl + self.reactors_lbl

    def init_nidx(self):
        # if 'multieres' in self.opts.energy_model:
            # self.subdetectors_names = ['subdet%03i'%i for i in range(5)]
        # else:
        self.subdetectors_names = ()

        self.nidx = [
            ('d', 'detector',    ['juno']),
            ['r', 'reactor',     self.reactors],
            ['i', 'isotope',     self.isotopes],
            ('rt', 'reactors_tao', ['TS1']),
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('s', 'subdetector', self.subdetectors_names),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] ),
            ('m', 'lsnl_method', ['subst', 'matrix'] ),
        ]
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
        # lsnlsubst = keeplsnl and self.opts.lsnl=='subst'
        # lsnlmatrix = keeplsnl and self.opts.lsnl=='matrix'
        lsnlsubst=keeplsnl
        lsnlmatrix=False
        print("Energy model:", lsnlsubst and 'LSNL subst,' or lsnlmatrix and 'LSNL matrix,' or 'no LSNL,', keeperes and 'eres' or 'no eres')
        formula_options = dict(
                #
                # Oscillation probability
                #
                oscprob = {
                    'vacuum':        'sum[c]| pmns[c]*oscprob[c,d,r](enu())' ,
                    'matter':        'oscprob_matter[d,r](enu())',
                    'matter-approx': 'oscprob_msw_approx[d,r](enu())',
                    }[self.opts.oscprob],
                #
                # Geo neutrino
                #
                # geonu_spectrum = '+geonu_scale*bracket(geonu_norm_U238*geonu_spectrum_U238(enu_matrix()) + geonu_norm_Th232*geonu_spectrum_Th232(enu_matrix()))'
                                 # if 'geo' in self.opts.bkg else '',
                #
                # Energy model
                #
                lsnlmatrix = 'lsnl|',
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
                'livetime=bracket(daq_years*seconds_in_year)',
                'livetime_tao=bracket(daq_years_tao*seconds_in_year)',
                'conversion_factor',
                'efflivetime = bracket(eff * livetime)',
                'efflivetime_tao = bracket(eff_tao * livetime_tao)',
                # 'geonu_scale = eff * livetime[d] * conversion_factor * target_protons',
                #
                # Integration energy
                #
                'evis_hist_quench=evis_hist_quench()',
                'evis_hist_noquench=evis_hist_noquench()',
                #
                # Energy model
                #
                # JUNO substitution and matrix
                'lsnl_coarse = sum[l]| lsnl_weight[l] * lsnl_component_y[l]()',
                'lsnl_gradient = sum[l]| lsnl_weight[l] * lsnl_component_grad[l]()',
                'lsnl_interpolator| lsnl_x(), lsnl_coarse, evis_edges_quench(), evis() ',
                'lsnl_interpolator_grad| lsnl_gradient',
                'lsnl_edges| evis_hist_quench',
                # TAO matrix
                'lsnl_coarse_tao = sum[l]| lsnl_weight_tao[l] * lsnl_component_y_tao[l]()',
                'lsnl_interpolator_tao| lsnl_x_tao(), lsnl_coarse_tao, evis_edges_noquench() ',
                'lsnl_edges_tao| evis_hist_noquench',
                # Eres
                'eres_pars',
                'binning_sigma_correction_coarse_centers| sigma_correction_coarse()',
                'edep_of_evis_for_sigma_correction_base| lsnl_coarse, binning_smear_centers(evis_hist_quench)',
                'edep_of_evis_for_sigma_correction| lsnl_x()',
                'sigma_correction_fine| sigma_correction_coarse()',
                'sigma_correction_fine_base| binning_sigma_correction_coarse_centers(), edep_of_evis_for_sigma_correction()',
                'eres_sigmarel_default=sigma_correction_fine()*eres_sigmarel| binning_smear_centers()' if not self.opts.consistency_eres else '',
                'eres_sigmarel_default=eres_sigmarel| binning_smear_centers()'                         if self.opts.consistency_eres else '',
                # Alt resolution
                'eres_alt_fine_base| eres_alt_e_coarse(), binning_smear_centers()' if self.opts.override_eres else '',
                'eres_alt_fine| eres_alt_rel_coarse()' if self.opts.override_eres else '',
                # Final resolution
                'eres_sigmarel_input| eres_sigmarel_default' if not self.opts.override_eres else '',
                'eres_sigmarel_input| eres_alt_fine()'         if self.opts.override_eres else '',
                'eres_matrix| evis_hist_quench' if keeperes else '',
                #
                # Neutrino energy
                #
                'ctheta_switch| ctheta_zero(ctheta()), ctheta()',
                'ctheta_mesh_switch| ctheta_mesh_zero(ctheta()), ctheta()',
                'evis_all[m](lsnl_evis(), evis())',
                'enu| ee(evis_all()), ctheta_switch()',
                #
                # Oscillations
                #
                'pmns[c]',
                #
                # Worst case spectral distortions
                #
                'pmns_wc_no[c]',
                'pmns_wc_io[c]',
                'baseline_wc',
                'Oscprob_wc_full_no = sum[c]| pmns_wc_no[c]*oscprob_wc_no[c](enu())',
                'Oscprob_wc_full_io = sum[c]| pmns_wc_io[c]*oscprob_wc_io[c](enu())',
                'SpectralDistortion = Oscprob_wc_full_io/Oscprob_wc_full_no',
                #
                # Reactor part
                #
                'thermal_power_scale[r]', # should be loaded before baseline
                'duty_cycle[r]',
                'baseline[d,r]',
                'numerator = thermal_power_scale[r] * thermal_power_nominal[r] * '
                             'fission_fractions_scale[r,i] * fission_fractions_nominal[r,i]() * '
                             'conversion_factor',
                'isotope_weight = energy_per_fission_scale[i] * energy_per_fission[i] * fission_fractions_scale[r,i]',
                'energy_per_fission_avg = sum[i]| isotope_weight * fission_fractions_nominal[r,i]()',
                'power_livetime_factor = numerator / energy_per_fission_avg',
                # 'anuspec_unc_corrected|enu_juno, anuspec_unc_scale[i]()*anuspec[i](enu_unc())',
                'anuspec_unc_corrected=anuspec[i](enu())',
                #
                # Nominal reactor spectrum for SNF and offeq
                #
                'energy_per_fission_avg_nominal = sum[i] | energy_per_fission[i] * fission_fractions_nominal[r,i]()',
                'power_livetime_factor_nominal = conversion_factor * thermal_power_nominal[r] * fission_fractions_nominal[r,i]() / energy_per_fission_avg_nominal',
                'nominal_spec_per_reac| sum[i]| power_livetime_factor_nominal*anuspec_unc_corrected',
                #
                # SNF
                #
                'snf_fraction_coarse_x()',
                'snf_fraction_coarse_y()',
                'snf_fine_base[m]| snf_fraction_coarse_x(), enu()',
                'snf_fine[m]| snf_fraction_coarse_y()',
                'snf_in_reac = snf_norm * nominal_spec_per_reac() * snf_fine()',
                #
                # Offequilibrium
                #
                'offeq_fraction_coarse_x()',
                'offeq_fraction_coarse_y()',
                'offeq_fine_base[m]| offeq_fraction_coarse_x(), enu()',
                'offeq_fine[m]| offeq_fraction_coarse_y()',
                'offeq_in_reac = offeq_scale * nominal_spec_per_reac() * offeq_fine()',
                #
                # Backgrounds in JUNO
                #
                'acc_num            = r_frac_acc * days_in_second * livetime * acc_rate           * acc_rate_norm',
                'fastn_num          = r_frac     * days_in_second * livetime * fastn_rate         * fastn_rate_norm',
                'alphan_num         = r_frac     * days_in_second * livetime * alphan_rate        * alphan_rate_norm',
                'lihe_num           = r_frac     * days_in_second * livetime * lihe_rate          * lihe_rate_norm',
                'geonu_num          = r_frac     * days_in_second * livetime * geonu_rate         * geonu_rate_norm',
                'reactors_elbl_num  = r_frac     * days_in_second * livetime * reactors_elbl_rate * reactors_elbl_rate_norm',
                #
                'geonu_spectrum = frac_Th232 * geonu_Th232_spectrum() + frac_U238 * geonu_U238_spectrum()',
                #
                'accidentals   = bracket| acc_num                              * rebin_acc[d]   | acc_spectrum()',
                'fastn         = bracket| fastn_num  * fastn_shape_nuisance()  * rebin_fastn[d] | fastn_spectrum()',
                'alphan        = bracket| alphan_num * alphan_shape_nuisance() * rebin_alphan[d]| alphan_spectrum()',
                'lihe          = bracket| lihe_num   * lihe_shape_nuisance()   * rebin_lihe[d]  | lihe_spectrum()',
                'geonu         = bracket| geonu_num  * geonu_shape_nuisance()  * rebin_geonu[d] | geonu_spectrum',
                'reactors_elbl = bracket| reactors_elbl_num * reactors_elbl_shape_nuisance() * rebin_reactors_elbl[d]| reactors_elbl_spectrum()',
                'bkg_juno      = bracket| accidentals + lihe + alphan + fastn + geonu + reactors_elbl' if not self.opts.disable_lbl else '',
                'bkg_juno      = bracket| accidentals + lihe + alphan + fastn + geonu' if self.opts.disable_lbl else '',
                'bkg_juno_noacc = bracket| lihe + alphan + fastn + geonu + reactors_elbl' if not self.opts.disable_lbl else '',
                'bkg_juno_noacc = bracket| lihe + alphan + fastn + geonu' if self.opts.disable_lbl else '',
                # 'bkg_shape_variance = bin_width_factor * bkgbin_widths(accidentals) * sumsq_snapshot| sumsq_bkg| lihe_bin2bin*lihe, fastn_bin2bin*fastn, alphan_bin2bin*alphan, geonu_bin2bin*geonu',
                #
                # Backgrounds in TAO
                #
                'acc_num_tao    = days_in_second * livetime_tao * acc_rate_tao    * acc_rate_norm_tao',
                'fastn_num_tao  = days_in_second * livetime_tao * fastn_rate_tao  * fastn_rate_norm_tao',
                'lihe_num_tao   = days_in_second * livetime_tao * lihe_rate_tao   * lihe_rate_norm_tao',
                #
                'accidentals_tao  = bracket| acc_num_tao   *                             rebin_acc_tao        | acc_spectrum_tao()',
                'lihe_tao         = bracket| lihe_num_tao  * lihe_shape_nuisance_tao() * rebin_lihe_tao       | lihe_spectrum_tao()',
                'fastn_tao        = bracket| fastn_num_tao *                             rebin_fastn_tao      | fastn_spectrum_tao()',
                'fastn_sigma_tao  = bracket| fastn_num_tao *                             rebin_fastn_sigma_tao| fastn_shape_unc_rel()*fastn_spectrum_tao()',
                'bkg_tao          = bracket| accidentals_tao + lihe_tao + fastn_tao + fastn_sigma_tao*fastn_shape_corr_scale()',
                # 'bkg_shape_variance_tao = bin_width_factor * bkgbin_tao_widths(accidentals_tao) * sumsq_snapshot_tao| sumsq_bkg_tao| lihe_bin2bin*lihe_tao, fastn_bin2bin_tao*fastn_sigma_tao',
                #
                # IBD part
                #
                '''anuspec_rd_full = reactor_active_norm
                                        *( sum[i]| power_livetime_factor*anuspec_unc_corrected )
                                        + offeq_in_reac
                                        + snf_in_reac
                                        ''',
                'bump_correction_base[m]| bump_correction_coarse_centers(bump_correction_coarse()), enu()',
                'bump_correction[m]| bump_correction_coarse()',
                'bump_switch| bump_095(bump_correction()), bump_correction()',
                '''cross_section = bracket(
                                    ibd_xsec(enu(), ctheta_mesh_switch())*
                                    jacobian(enu(), ee(), ctheta_switch())*
                                    bump_switch()
                                )''',
                # 'cross_section_subst=select1[m, "subst"]|cross_section',
                'cross_section_matrix=select1[m, "matrix"]|cross_section',
                '''ibd_juno_all=r_frac*norm_reac*norm_juno*anue_spectrum_scale_final()* rebin|
                         {eres} {lsnlmatrix}
                           rebin_juno_internal |
                               kinint2_juno(
                                   DistortSpectrum(
                                       sum[r](
                                           selective_jacobian(
                                               (
                                                   (baselineweight[r,d]*efflivetime*target_protons*duty_cycle[r])
                                                   * anuspec_rd_full
                                                   * {oscprob}
                                               )*cross_section
                                               ,
                                               lsnl_interpolator_grad()
                                           )
                                       ),
                                       SpectralDistortion
                                    )
                               )
                '''.format(**formula_options),
                #
                # Total observation
                #
                'observation_juno=ibd_juno_all + bkg_juno',
                # 'variance_juno_all = staterr2(observation_juno) + bkg_shape_variance',
                #
                # TAO part
                #
                'eres_matrix_tao| evis_hist_noquench',
                'anuspec_rd_full_tao_all = select1[r,"TS1"]| anuspec_rd_full',
                'anuspec_rd_full_tao = select1[m,"matrix"]| anuspec_rd_full_tao_all',
                'SpectralDistortion_tao = select1[m,"matrix"]| SpectralDistortion',
                '''ibd_tao = norm_reac*norm_tao*rebin_tao| eres_tao|
                               {lsnl_tao}
                                 eleak_tao|
                                   rebin_tao_internal|
                                       kinint2_tao|
                                         DistortSpectrumTAO(
                                           sum[rt](
                                             bracket(
                                                (baselineweight_tao[rt]*efflivetime_tao*target_protons_tao*duty_cycle_tao)
                                               * anuspec_rd_full_tao * cross_section_matrix)),
                                           SpectralDistortion_tao
                                         )
                             '''.format(**formula_options),
                #
                # TAO total observation
                #
                'observation_tao=ibd_tao + bkg_tao',
                # 'variance_tao = staterr2_tao(observation_tao) + bkg_shape_variance_tao',
                #
                # Comparison cases
                #
                'cmp_accidentals = acc_num    * acc_spectrum()',
                'cmp_fastn       = fastn_num  * fastn_spectrum()',
                'cmp_alphan      = alphan_num * alphan_spectrum()',
                'cmp_lihe        = lihe_num   * lihe_spectrum()',
                'cmp_geonu       = geonu_num  * geonu_spectrum',
                'cmp_reactors_elbl = reactors_elbl_num * reactors_elbl_spectrum()',
                'cmp_bkg_juno    = cmp_rebin_bkg[d]| cmp_accidentals + cmp_lihe + cmp_alphan + cmp_fastn + cmp_geonu + cmp_reactors_elbl',
                'cmp_juno_case12 = cmp_rebin_12| norm_reac*norm_juno*rebin_juno_internal()',
                'cmp_juno_case3  = cmp_rebin_3| norm_reac*norm_juno*eres()',
                'cmp_juno_case4  = cmp_rebin_4(cmp_juno_case3)+cmp_bkg_juno',
                #
                # Data
                #
                'rebin_data_juno'
                ]

    def parameters(self):
        ns = self.namespace
        dmxx = 'pmns.DeltaMSq13'
        for par in [dmxx, 'pmns.SinSqDouble12', 'pmns.DeltaMSq12']:
            ns[par].setFree()

        if self.opts.enable_hz:
            hz = ns['thermal_power_nominal.HZ']
            hz.set(hz.value()*0.5)

    def init_configuration(self):
        step = 0.005
        if self.opts.fine_step:
            step = self.opts.fine_step*1.e-3
        step_tao = 0.01
        edges_internal = np.arange(0.0, 12.001, step)
        edges_internal_tao = np.arange(0.0, 12.001, step_tao)
        juno_edges = np.concatenate( (
                                    [0.8],
                                    np.arange(0.94, 7.44, 0.02),
                                    np.arange(7.44, 7.8, 0.04),
                                    np.arange(7.8, 8.2, 0.1),
                                    [8.2, 12.0]
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

        _, enu_edges = self.init_tao_binning()
        anue_spec_unc_edges = enu_edges[(enu_edges>0.0)*(enu_edges<=8.0)]
        anue_spec_unc_edges = np.arange(anue_spec_unc_edges[0], anue_spec_unc_edges[-1]+1.e-6, 0.01)
        anue_spec_unc_edges = np.concatenate([anue_spec_unc_edges, enu_edges[enu_edges>8.0]])
        if self.opts.comparison:
            print('Running in comparison mode! Enu and unc bins modified')
            a, b, c = enu_edges[enu_edges<1.99], enu_edges[(enu_edges>=1.99)*(enu_edges<8.01)], enu_edges[enu_edges>=8.01]
            enu_edges = np.concatenate([a, np.arange(2.0, 8.01, 0.250), c]).astype('d')
            a, b, c = anue_spec_unc_edges[anue_spec_unc_edges<1.99], anue_spec_unc_edges[(anue_spec_unc_edges>=1.99)*(anue_spec_unc_edges<8.01)], anue_spec_unc_edges[anue_spec_unc_edges>=8.01]
            anue_spec_unc_edges = np.concatenate([a, np.arange(2.0, 8.01, 0.010), c]).astype('d')

        from physlib import pc
        energy_offset = pc.DeltaNP - pc.ElectronMass
        enu_threshold=1.8
        EvisToEnu = lambda Evis: Evis + energy_offset

        tao_edges = np.concatenate([juno_edges[juno_edges<7.999], [8.0, 9.0, 12.0]])
        tao_edges[0] = 0.9
        enu_edges1 = EvisToEnu(tao_edges[tao_edges>=8.0])
        enu_edges = np.concatenate([enu_edges[enu_edges<8.0], enu_edges1[:-1]])

        if self.opts.verbose:
            print('JUNO binning:', format_binning(juno_edges, 2))
            print('TAO binning:', format_binning(tao_edges, 2))
            print('Enu binning:',  format_binning(enu_edges, 4))
            print('Enu spec. unc. binning',  format_binning(anue_spec_unc_edges, 4))

        common_input_location='data/data_juno/data-joint/2020-06-11-NMO-Analysis-Input'
        common_input_root_file='JUNOInputs2021_05_28_upd.root'
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
                    pars = f'{common_input_location}/files/juno_input_numbers_v3.py',
                    skip = ('duty_cycle',),
                    state= 'fixed'
                    ),
                numbers1 = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/juno_target_mass_v1.py',
                    state= 'fixed'
                    ),
                bkg_shape_unc = dict(
                    bundle = dict(name='parameters', version='v06'),
                    pars = f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
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
                    #
                    # The reactor_data_lbl subconfiguration will also populate thermal power and duty_cycle for LBL reactors
                    # Thus, baseline and baselineweight should be triggered after thermal_power and duty_cycle
                    #
                    bundle = dict(name="bundles_combination_1d", version = "v01", major='r'),
                    slices = {
                        'mbl': self.reactors_mbl.copy(),
                        'lbl': 'rest'
                        },
                    bundles = dict(
                        mbl = dict(
                            baselines_juno_mbl = dict(
                                bundle = dict(name='reactor_baselines', version='v02', major = 'rd'),
                                reactors  = f'{common_input_location}/files/reactor_baselines_v2.yaml',
                                reactors_key = 'reactor_baseline',
                                detectors = dict(juno=0.0),
                                unit = 'km'
                                ),
                            ),
                        lbl = dict(
                            reactor_data_lbl = dict(
                                bundle = dict(name='reactor_data', version='v01', major='r'),
                                filename  = f'{common_input_location}/files/thermal_power_world_below2000_v01.dat',
                                units = { 'distance': 'km', 'power': 'mw' },
                                columns = ['name', 'power', 'distance', 'duty_cycle'],
                                relative_uncertainty = 0.005,
                                ),
                            )
                    ),
                    permit_empty_rest=True,
                ),
                #
                # Reactor
                #
                thermal_power_juno = dict(
                    bundle = dict(name="bundles_combination_1d", version = "v01", major='r'),
                    slices = {
                        'mbl': self.reactors_mbl.copy(),
                        # 'lbl': 'rest'
                        },
                    bundles = dict(
                        mbl = dict(
                            thermal_power_mbl = dict(
                                bundle = dict(name="parameters", version = "v06", names=dict(thermal_power='thermal_power_nominal')),
                                pars = f'{common_input_location}/files/thermal_power_v2.yaml',
                                separate_uncertainty = '{}_scale'
                                ),
                            duty_cycle_mbl = dict(
                                bundle = dict(name='parameters', version='v06', major=''),
                                pars = f'{common_input_location}/files/juno_input_numbers_v3.py',
                                skip = ('daq_years', 'global_norm', 'eff'),
                                state= 'fixed'
                                ),
                            ),
                    ),
                    permit_empty_rest=True,
                    permit_unprocessed_rest=True,
                ),
                energy_per_fission =  dict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = f'{common_input_location}/files/energy_per_fission.yaml',
                        separate_uncertainty = '{}_scale'
                        ),
                reactor_spectrum_shape_unc = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'anue_spectrum_scale_final',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'root': f'{common_input_location}/{common_input_root_file}',
                            'hist': 'TAOUncertainty',
                            },
                        # fixed: see self.opts.combination
                        ),
                #
                # Backgrounds
                #
                bkg_rate = dict(
                        bundle = dict(name="parameters", version = "v06"),
                        pars = f'{common_input_location}/files/bkg_rates_v6.yaml',
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
                    bundle    = dict(name='integral_2d1d', version='v06'),
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
                        # 'kinint2_tao': {'label': 'TAO integral', 'noindex': True},
                        'kinint2_tao': {'label': 'TAO integral', 'index': ()},
                        }
                    ),
                energies = dict(
                        bundle = dict(name='arrange', version='v01'),
                        names = 'evis_all'
                        ),
                rebin_internal_juno = dict(
                    bundle = dict(name='rebin', version='v05', major='', names={'rebin_hist': 'evis_hist_quench', 'rebin_points': 'evis_edges_quench'}),
                    rounding = 6,
                    edges = edges_internal,
                    instances={
                        'rebin_juno_internal': 'JUNO internal binning',
                        }
                    ),
                rebin_internal_tao = dict(
                    bundle = dict(name='rebin', version='v05', major='', names={'rebin_hist': 'evis_hist_noquench', 'rebin_points': 'evis_edges_noquench'}),
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
                    instances={
                        'rebin': 'JUNO final',
                        }
                    ),
                rebin_data_juno = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = juno_edges,
                    instances={
                        'rebin_data_juno': 'JUNO data'
                        }
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
                        'rebin_reactors_elbl': 'Reactors L\\>2000 km {autoindex}',
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
                        'rebin_fastn_sigma_tao': 'Fast neutrons corr. TAO',
                        }
                    ),
                rebin_tao = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = tao_edges,
                    instances={ 'rebin_tao': 'TAO final' }
                    ),
                rebin_comparison_bkg = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = np.arange(0.8, 12.0+1.e-6, 0.02),
                    instances={
                        'cmp_rebin_bkg': 'Rebin comp. bkg.',
                        }
                    ),
                rebin_comparison = dict(
                    bundle = dict(name='rebin', version='v04', major=''),
                    rounding = 5,
                    edges = np.arange(0.8, 12.0+1.e-6, 0.02),
                    instances={
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
                #
                # Background shape variance (old)
                #
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
                    bundle     = dict(name='detector_energy_leakage_root', version='v02', names=lambda s: s+'_tao'),
                    filename   = f'{common_input_location}/{common_input_root_file}',
                    matrixname = 'TAO_response_matrix_25',
                    matrix_is_upper = False
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
                ibd_xsec_vogel = dict(
                    bundle = dict(name='xsec_ibd', version='v05', major='m'),
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
                        fixed = False,                                       # see also: self.opts.worst_case_fit_pars
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
                        formula = 'khan-approx',
                        density=(2.45, 0.15),
                        electron_density=0.5
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
                    state='fixed',                                           # see also: self.opts.worst_case_fit_pars
                    labels=dict(
                        baseline_wc = 'Baseline for worst case distortion, km',
                        ),
                    pars =  {'baseline_wc': 52.514568468905615},
                    ),
                oscpars_wc_no = dict(
                        bundle = dict(name='oscpars_ee', version='v01', names={'pmns': 'pmns_wc_no'}),
                        fixed = True,                                        # see also: self.opts.worst_case_fit_pars
                        parameters = dict(
                            DeltaMSq13    = 0.0025332742137640567,
                            DeltaMSq12    = 7.526728300087154e-05,
                            SinSqDouble12 = 0.8506780662379546,
                            SinSqDouble13 = 0.0599399133630274,
                            )
                        ),
                oscpars_wc_io = dict(
                        bundle = dict(name='oscpars_ee', version='v01', names={'pmns': 'pmns_wc_io'}),
                        fixed = True,                                     # see also: self.opts.worst_case_fit_pars
                        parameters = dict(
                            DeltaMSq13    = 0.0024808313216080047,
                            DeltaMSq12    = 7.527218361676647e-05,
                            SinSqDouble12 = 0.850712052757394,
                            SinSqDouble13 = 0.05975399333523377,
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
                # anuspec_hm = dict(
                        # bundle = dict(name='reactor_anu_spectra', version='v06'),
                        # name = 'anuspec',
                        # filename = f'{common_input_location}/{common_input_root_file}',
                        # objectnamefmt = 'HuberMuellerFlux_{isotope}',
                        # spectral_parameters='fixed',
                        # varmode=self.opts.free_pars_mode or 'log',
                        # varname='anue_weight_{index:04d}',
                        # ns_name='spectral_weights',
                        # edges=enu_edges
                        # ),
                anuspec_hm = dict(
                        bundle = dict(name='reactor_anu_spectra', version='v07', major='i'),
                        name = 'anuspec',
                        filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                                    'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
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
                reactor_active_norm = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'reactor_active_norm',
                        label='Reactor nu (active norm)',
                        pars = uncertain(1.0, 'fixed'),
                        ),
                #
                # Offequilibrium
                #
                offeq_correction_coarse = dict(
                        bundle = dict(name='load_graph', version='v01'),
                        name = 'offeq_fraction_coarse',
                        filenamefmt = f'{common_input_location}/{common_input_root_file}',
                        objectnamefmt = 'NonEq_FluxRatio',
                        verbose = True,
                        labelx = 'Offeq E coarse',
                        labely = 'Offeq Y (fraction) coarse',
                        ),
                offeq_fine = dict(
                        bundle = dict(name='interpolation_1d', version='v01', major='m'),
                        name='offeq_fine',
                        kind='linear',
                        strategy=('extrapolate', 'constant'),
                        label='Offequilibrium idx|{autoindex}',
                        labelfmt='Offequilibrium fine|{autoindex}',
                        ),
                offeq_scale =  dict(
                        bundle = dict(name="parameters", version = "v06"),
                        labels={
                            'offeq_scale': 'Offeq. scale',
                            },
                        pars =  {
                            'offeq_scale': (1.0, 30),
                            'uncertainty_mode': 'percent',
                            },
                        ),
                #
                # SNF
                #
                snf_correction_coarse = dict(
                        bundle = dict(name='load_graph', version='v01'),
                        name = 'snf_fraction_coarse',
                        filenamefmt = f'{common_input_location}/{common_input_root_file}',
                        objectnamefmt = 'SNF_FluxRatio',
                        verbose = True,
                        labelx = 'SNF E coarse',
                        labely = 'SNF Y (fraction) coarse',
                        ),
                snf_fine = dict(
                        bundle = dict(name='interpolation_1d', version='v01', major='m'),
                        name='snf_fine',
                        kind='linear',
                        strategy=('extrapolate', 'constant'),
                        label='SNF idx|{autoindex}',
                        labelfmt='SNF fine|{autoindex}',
                        ),
                snf_snaphost_spectrum = dict(
                    bundle = dict(name='trans_snapshot', version='v01', major=''),
                    instances={'nominal_spec_per_reac': 'Reactor nominal spectrum|{autoindex} (for SNF/Offeq)'}
                    ),
                snf_norm = dict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = 'snf_norm',
                        label='SNF norm',
                        pars = uncertain(1.0, 30, 'percent'),
                        ),
                #
                # Bump correction
                #
                bump_correction = dict(
                        bundle = dict(name='interpolation_1d', version='v01', major='m'),
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
                        bundle = dict(name='filllike', version='v01', major=()),
                        instances = {
                            'bump_095': 'n=0.95',
                            },
                        value = 0.95
                        ),
                bump_switch = dict(
                        bundle = dict(name='switch', version='v01', major=(), names={'condition': 'bump_switchvar'}),
                        instances = {
                            'bump_switch': '{{n=0.95 or bump|{autoindex}}}',
                            },
                        varlabel = 'bump switch (n=0.95 or bump)',
                        ninputs = 2,
                        default = 1
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
                ##
                ## LSNL
                ##
                lsnl = dict(
                        bundle = dict(name='energy_nonlinearity_db_root_subst', version='v02', major='l',),
                        names      = {
                            'nominal': 'positronScintNL',
                            'pull0': 'positronScintNLpull0',
                            'pull1': 'positronScintNLpull1',
                            'pull2': 'positronScintNLpull2',
                            'pull3': 'positronScintNLpull3',
                            },
                        filename   = f'{common_input_location}/{common_input_root_file}',
                        nonlin_range = (0.5, 12.),
                        supersample=20,
                        expose_matrix = False,
                        minor_extra = 'm',
                        bypass_minor = 'juno.subst',
                        verbose = 0
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
                        nonlin_range = (0.499, 12.),
                        extrapolate_range = (1.0, 12.),
                        supersample=20,
                        expose_matrix = False
                        ),
                selective_jacobian=dict(
                        bundle = dict(name='selective_ratio', version='v01', major='m'),
                        name='selective_jacobian',
                        substring_skip='matrix',
                        broadcast=True,
                        labelfmt_ratio='Jacobian applied {autoindex}',
                        labelfmt_view='View (no Jacobian) {autoindex}'
                    ),
                ##
                ## Energy resolution
                ##
                eres_smearing = dict(
                        bundle = dict(name='detector_eres_inputsigma', version='v02', major=''),
                        expose_matrix = False
                        ),
                eres_pars = dict(
                        # pars: sigma_e/e = sqrt(a^2 + b^2/E + c^2/E^2),
                        # a - non-uniformity
                        # b - statistical term
                        # c - noise
                        bundle = dict(name='parameters', version = 'v07'),
                        pars = f'{common_input_location}/files/juno_eres_v3.yaml',
                        depth = 2
                    ),
                eres_sigmarel = dict(
                        bundle = dict(name='energy_resolution_sigmarel_abc', version='v01'),
                        parameter = 'eres_pars',
                        pars = ('a_nonuniform', 'b_stat', 'c_noise'),
                        ),
                sigma_correction_coarse = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/{common_input_root_file}',
                    formats = ['Eres_ratio_PMT_Model'],
                    names   = ['sigma_correction_coarse'],
                    labels  = ['σ correction (coarse)'],
                    normalize = False,
                    ),
                edep_of_evis_for_sigma_correction = dict(
                        bundle = dict(name='interpolation_1d', version='v01'),
                        name='edep_of_evis_for_sigma_correction',
                        kind='linear',
                        strategy='nearestedge',
                        label='Edep(Evis) (inverse LSNL) idx',
                        labelfmt='Edep(Evis) (inverse LSNL) for σ correction',
                        ),
                sigma_correction_fine = dict(
                        bundle = dict(name='interpolation_1d', version='v01'),
                        name='sigma_correction_fine',
                        kind='linear',
                        strategy='nearestedge',
                        label='σ correction idx',
                        labelfmt='σ correction (fine)',
                        ),
                bins_internal = dict(
                    bundle = dict(name='trans_histedges', version='v01'),
                    types = ('centers', ),
                    instances={
                        'binning_smear': 'E centers (smearing)',
                        'binning_sigma_correction_coarse': 'σ rel centers (coarse)'
                        }
                    ),
                #
                # Backgrounds
                #
                bkg_spectra = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/{common_input_root_file}',
                    formats = ['AccBkgHistogramAD',         'Li9BkgHistogramAD',            'FnBkgHistogramAD',            'AlphaNBkgHistogramAD',        'GeoNuTh232',                       'GeoNuU238'],
                    names   = ['acc_spectrum',              'lihe_spectrum',                'fastn_spectrum',              'alphan_spectrum',             'geonu_Th232_spectrum',             'geonu_U238_spectrum'],
                    labels  = ['Acc. JUNO|(norm spectrum)', '9Li/8He JUNO|(norm spectrum)', 'Fast n JUNO|(norm spectrum)', 'AlphaN JUNO|(norm spectrum)', 'GeoNu Th232 JUNO|(norm spectrum)', 'GeoNu U238 JUNO|(norm spectrum)'],
                    normalize = True,
                    ),
                bkg_spectra_reac = dict(
                    bundle    = dict(name='root_histograms_v05'),
                    filename  = f'{common_input_location}/{common_input_root_file}',
                    formats = ['OtherReactorSpectrum'],
                    names   = ['reactors_elbl_spectrum'],
                    labels  = ['Reactors L\\>2000 km|(norm spectrum)'],
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
                bkg_fnshape_unc_pars = dict(
                        bundle      = dict(name='vararray', version='v01'),
                        name        = 'fastn_shape_corr_scale',
                        parnamefmt  = '{name}.{name}_{i:03d}',
                        count       = len(tao_edges)-1,
                        central     = 0.0,
                        uncertainty = 1.0,
                        label       = 'Fast n. correction scale',
                        parlabelfmt = 'Fast n. corr. scale: bin {i: 3d} ({left:.3f}, {right:.3f})',
                        edges       = tao_edges
                        ),
                geonu_fractions=dict(
                        bundle = dict(name='var_fractions_v02'),
                        names = [ 'Th232', 'U238' ],
                        format = 'frac_{component}',
                        fractions = uncertaindict(
                            Th232 = ( 0.23, 'fixed' )
                            ),
                        ),
                #
                # Background shape variance (new)
                #
                bkg_shape_fastn = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'fastn_shape_nuisance',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'fastn_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        ),
                bkg_shape_lihe = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'lihe_shape_nuisance',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'lihe_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        ),
                bkg_shape_lihe_tao = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'lihe_shape_nuisance_tao',
                        mode = 'relative',
                        edges_target = tao_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'lihe_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        ),
                bkg_shape_alphan = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'alphan_shape_nuisance',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'alphan_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        ),
                bkg_shape_geonu = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'geonu_shape_nuisance',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'geonu_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        ),
                bkg_shape_reactors_elbl = dict(
                        bundle = dict(name='bin2bin_uncertainty', version='v01', major=()),
                        name = 'reactors_elbl_shape_nuisance',
                        mode = 'relative',
                        edges_target = juno_edges,
                        uncertainty={
                            'yaml': f'{common_input_location}/files/bkg_bin2bin_v2.yaml',
                            'uncertainty': 'geonu_bin2bin',
                            'binwidth': 'bin_width'
                            }
                        )
                )

        if self.opts.consistency_eres:
            print('\033[35mConsistency mode: previous eres\033[0m')
            self.cfg.eres_pars.pars = f'{common_input_location}/files/juno_eres_v2.yaml'

        if self.opts.worst_case_fit_pars:
            print('\033[35mWorst case fitting mode: fix osc pars, release baseline_wc and pmns_io\033[0m')
            self.cfg.oscpars['fixed']=True
            self.cfg.oscpars_wc_io['fixed']=False
            self.cfg.oscpars_wc_no['fixed']=False
            self.cfg.numbers_wc['state']='free'

        if self.opts.combination:
            print('\033[35mEnable combination mode\033[0m')
            self.cfg.reactor_spectrum_shape_unc.fixed=True

        self.cfg['r_frac'] = dict(
                bundle = dict(name='parameters', version='v05'),
                state='fixed',
                labels=dict(
                    r_frac     = 'Fraction of events due to R cut (not applies to accidental)',
                    ),
                pars =  dict(
                    r_frac = 1.0,
                    ),
                )
        self.cfg['r_frac_acc'] = dict(
                bundle = dict(name='parameters', version='v05'),
                state='fixed',
                labels=dict(
                    r_frac_acc = 'Fraction of accidental events due to R cut',
                    ),
                pars =  dict(
                    r_frac_acc = 1.0,
                    ),
                )
        if self.opts.override_eres:
            self.load_eres_alt(*self.opts.override_eres)

    def load_eres_alt(self, fname, reco, num):
        import pickle, sys
        from scipy.interpolate import interp1d

        num = int(num)
        print('\033[31mOverride energy resolution\033[0m')
        print(f'Read energy resolution {num} for {reco} from {fname}')

        with open(fname, 'rb') as f:
            data = pickle.load(f)

        datalist = data[reco]

        try:
            datum = datalist[num]
        except IndexError:
            print('override-eres: index out of bounds')
            sys.exit(1)

        eres_e       = np.array(datum['eres']['energies'])
        percent=0.01
        eres_res_rel = np.array(datum['eres']['resolution'])*percent
        eres_res_abs = eres_e*eres_res_rel

        eres_e       = np.array(datum['eres']['energies'])
        percent=0.01
        eres_res_rel = np.array(datum['eres']['resolution'])*percent
        # eres_res_abs = eres_e*eres_res_rel

        fine_e = np.linspace(eres_e[0], eres_e[-1], (len(eres_e)-1)*10+1)
        fine_res_rel = interp1d(eres_e, eres_res_rel, kind='quadratic')(fine_e)
        print(fine_e)
        print(fine_res_rel)

        self._eres_alt_e_coarse = C.Points(fine_e)
        self._eres_alt_rel_coarse = C.Points(fine_res_rel)

        self.cfg['eres_alt_e_coarse'] = dict(
                bundle = dict(name='predefined', version='v01'),
                name='eres_alt_e_coarse',
                inputs = None,
                outputs = self._eres_alt_e_coarse.single()
                )
        self.cfg['eres_alt_rel_coarse'] = dict(
                bundle = dict(name='predefined', version='v01'),
                name='eres_alt_rel_coarse',
                inputs = None,
                outputs = self._eres_alt_rel_coarse.single()
                )
        self.cfg['eres_alt_fine'] = dict(
                bundle = dict(name='interpolation_1d', version='v01'),
                name='eres_alt_fine',
                kind='linear',
                strategy=('extrapolate', 'extrapolate'),
                label='Energy resolution from ML fine',
                labelfmt='Energy resolution from ML fine',
                )

        self.cfg['r_frac']['state']='free'
        self.cfg['r_frac']['pars']['r_frac'] = datum['frac']['events']
        self.cfg['r_frac_acc']['pars']['r_frac_acc'] = datum['frac']['accidentals']
        # r_frac_acc is kept fixed

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

    # def update_variance(self):
        # snapshots = [s for provider in (self.context.providers[name] for name in ['sumsq_snapshot', 'sumsq_snapshot_tao']) for s in provider.bundle.objects]
        # outputs = self.context.outputs

        # print('Compute and fix JUNO and TAO variance (stat, bkg)')
        # for snapshot in snapshots:
            # snapshot.nextSample()
        # outputs.variance_juno.juno.touch()
        # # outputs.variance_tao.touch()

    def register(self):
        from gna.env import env
        futurens = env.future.child(('spectra', self.namespace.name))

        k0 = ('extra',)
        for k, v in self.context.outputs.items(nested=True):
            futurens[k0+k]=v

        outputs = self.context.outputs
        inputs = self.context.inputs

        # futurens[('variance', 'juno', 'stat')]     = outputs.staterr2.juno
        # futurens[('variance', 'juno', 'bkgshape')] = outputs.bkg_shape_variance.juno
        # futurens[('variance', 'juno', 'full')]     = outputs.variance_juno.juno
        # futurens[('variance', 'tao', 'stat')]      = outputs.staterr2_tao
        # futurens[('variance', 'tao', 'bkgshape')]  = outputs.bkg_shape_variance_tao
        # futurens[('variance', 'tao', 'full')]      = outputs.variance_tao
        # Force calculation of the stat errors

        # env.future[('hooks', self.namespace.name, 'variance')]=lambda: self.update_variance()

        futurens[('juno', 'initial')]  = StripNestedDict(outputs.rebin_juno_internal.juno)
        try:
            futurens[('juno', 'equench')] = StripNestedDict(outputs.lsnl.juno)
            futurens[('juno', 'fine')] = StripNestedDict(outputs.lsnl.juno)
        except: pass
        try:
            futurens[('juno', 'evis')] = StripNestedDict(outputs.eres.juno)
            futurens[('juno', 'fine')] = StripNestedDict(outputs.eres.juno)
        except: pass
        futurens[('juno', 'rebin')]    = StripNestedDict(outputs.rebin.juno)
        futurens[('juno', 'bkg')]          = outputs.bkg_juno
        futurens[('juno', 'ibd')]          = outputs.ibd_juno_all.juno.subst
        futurens[('juno', 'ibd_matrix')]   = outputs.ibd_juno_all.juno.matrix
        futurens[('juno', 'final')]        = outputs.observation_juno.juno.subst
        futurens[('juno', 'final_matrix')] = outputs.observation_juno.juno.matrix

        futurens[('juno', 'data_in')]  = inputs.rebin_data_juno['00']
        futurens[('juno', 'data')]     = outputs.rebin_data_juno

        futurens[('tao', 'initial')]   = outputs.rebin_tao_internal
        try:
            futurens[('tao', 'edep')]  = outputs.eleak_tao
            futurens[('tao', 'fine')]  = outputs.eleak_tao
        except: pass
        try:
            futurens[('tao', 'equench')] = outputs.lsnl_tao
            futurens[('tao', 'fine')]    = outputs.lsnl_tao
        except: pass
        try:
            futurens[('tao', 'evis')]  = outputs.eres_tao
            futurens[('tao', 'fine')]  = outputs.eres_tao
        except: pass
        futurens[('tao', 'rebin')]     = outputs.rebin_tao
        futurens[('tao', 'ibd')]       = outputs.ibd_tao
        futurens[('tao', 'final')]     = outputs.observation_tao

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.rebin.juno.subst
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    def init_tao_binning(self):
        mode, step = self.opts.binning_tao.split('-')
        step = int(step)
        fname = f'data/local/tao-binning-v05b-{step}.dat'
        evis, enu, _ = np.loadtxt(fname, unpack=True)
        enu=enu[enu!=0.0]

        # Use wide first bin
        enu = np.concatenate(((enu[0],), enu[enu>=enu[0]+0.04]))

        if mode=='const':
            step_mev = step/1000.0
            enu = np.concatenate( ( (enu[0],), np.arange(enu[1], 7.44, step_mev), enu[enu>=7.43999]))

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
