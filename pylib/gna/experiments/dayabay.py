from gna.exp import baseexp
from gna.configurator import uncertaindict, uncertain, NestedDict
from gna.expression.index import NIndex
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.bundle import execute_bundles

from itertools import chain
import numpy as N
import ROOT as R

seconds_per_day = 60.*60.*24.
percent = 0.01
nominal_mass = 20000.
snf_energy_per_decay = 1./(0.563*202.36 + 0.079*205.99 + 0.301*211.12 + 0.057*214.26)


class exp(baseexp):
    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('-e', '--embed', action='store_true', help='embed')
        parser.add_argument('-c', '--composition', default='complete', choices=['complete', 'minimal', 'small'], help='Set the indices coverage')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
        parser.add_argument('--stats', action='store_true', help='print stats')
        parser.add_argument('--no-snf', action='store_true', help='Disable SNF')
        parser.add_argument('--lihe-fractions', default='no', choices=['no', 'common', 'independent'],
                            help='Split the data over two time windows after muon veto to '
                            'allow suppression of Li/He background')

    def init(self):
        self.formula = dict()

        self.init_nidx()
        self.define_topology()
        self.define_labels()
        self.init_configuration()
        self.build()
        try:
            self.register()
        except:
            pass

        if self.opts.stats:
            self.print_stats()

    def init_nidx(self):
        self.detectors = ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']
        self.reactors  = ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']
        self.nidx = [
            ('s', 'site',        ['EH1', 'EH2', 'EH3']),
            ('d', 'detector',    self.detectors,
                                 dict(short='s', name='site', map=dict([('EH1', ('AD11', 'AD12')), ('EH2', ('AD21', 'AD22')), ('EH3', ('AD31', 'AD32', 'AD33', 'AD34'))]))),
            ('r', 'reactor',     self.reactors),
            ('i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']),
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'] ),
            ('w', 'time_window', ['A', 'B'] )
        ]
        if self.opts.composition=='minimal':
            self.nidx[0][2][:] = self.nidx[0][2][:1]
            self.nidx[1][2][:] = self.nidx[1][2][:1]
            self.nidx[2][2][:] = self.nidx[2][2][:1]
            self.nidx[3][2][:] = self.nidx[3][2][:1]
        elif self.opts.composition=='small':
            self.nidx[0][2][:] = self.nidx[0][2][:2]
            self.nidx[1][2][:] = ['AD11']
            self.nidx[2][2][:] = ['DB1']
            self.nidx[3][2][:] = ['U235']
        self.nidx = NIndex.fromlist(self.nidx)

        self.groups=NestedDict(
                exp  = { 'dayabay': self.detectors },
                det  = { d: (d,) for d in self.detectors },
                site = NestedDict([
                    ('EH1', ['AD11', 'AD12']),
                    ('EH2', ['AD21', 'AD22']),
                    ('EH3', ['AD31', 'AD32', 'AD33', 'AD34']),
                    ]),
                adnum_local = NestedDict([
                    ('1', ['AD11', 'AD21', 'AD31']),
                    ('2', ['AD12', 'AD22', 'AD32']),
                    ('3', ['AD33']),
                    ('4', ['AD34']),
                    ]),
                adnum_global = NestedDict([
                    ('1', ['AD11']), ('2', ['AD12']),
                    ('3', ['AD21']), ('8', ['AD22']),
                    ('4', ['AD31']), ('5', ['AD32']), ('6', ['AD33']), ('7', ['AD34']),
                    ]),
                adnum_global_alphan_subst = NestedDict([
                    ('1', ['AD11']), ('2', ['AD12']),
                    ('3', ['AD21', 'AD22']),
                    ('4', ['AD31']), ('5', ['AD32']), ('6', ['AD33', 'AD34']),
                    ])
                )

    def init_configuration(self):
        self.cfg = NestedDict(
            integral = NestedDict(
                bundle   = dict(name='integral_2d1d', version='v03'),
                variables = ('evis', 'ctheta'),
                edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
                xorders   = 4,
                yorder   = 2,
                ),
            ibd_xsec = NestedDict(
                bundle = dict(name='xsec_ibd', version='v02'),
                order = 1,
                pdg_year='dyboscar'
                ),
            oscprob = NestedDict(
                bundle = dict(name='oscprob', version='v04', major='rdc'),
                #  pdg_year='dyboscar',
                dm='23',
                ),
            anuspec = NestedDict(
                bundle = dict(name='reactor_anu_spectra', version='v04'),
                name = 'anuspec',
                filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                            'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                free_params=True, # enable free spectral model
                varmode='plain',
                varname='anue_weight_{index}',
                ns_name='spectral_weights',
                edges = N.concatenate( ( N.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
                ),
            offeq_correction = NestedDict(
                bundle = dict(name='reactor_offeq_spectra',
                              version='v04', major='ir'),
                offeq_data = 'data/reactor_anu_spectra/Mueller/offeq/mueller_offequilibrium_corr_{isotope}.dat',
                ),
            eff = NestedDict(
                bundle = dict(name='efficiencies', version='v02',
                    names = dict(
                        norm = 'global_norm'
                        ),
                    major=''
                    ),
                correlated   = False,
                uncorrelated = True,
                norm         = True,
                efficiencies = 'data/dayabay/efficiency/P15A_efficiency.py'
                ),
            livetime = NestedDict(
                bundle = dict(name='dayabay_livetime_hdf', version='v02'),
                file   = 'data/dayabay/data/P15A/dubna/dayabay_data_dubna_v15_bcw_adsimple.hdf5',
            ),
            baselines = NestedDict(
                bundle = dict(name='reactor_baselines', version='v01', major='rd'),
                reactors  = 'data/dayabay/reactor/coordinates/coordinates_docDB_9757.py',
                detectors = 'data/dayabay/ad/coordinates/coordinates_docDB_9757.py',
                unit = 'm'
                ),
            thermal_power = NestedDict(
                    bundle = dict(name='dayabay_reactor_burning_info_v02', major='ri'),
                    reactor_info = 'data/dayabay/reactor/power/WeeklyAvg_P15A_v1.txt.npz',
                    fission_uncertainty_info = 'data/dayabay/reactor/fission_fraction/2013.12.05_djurcic.py',
                    add_ff = True,
                    nominal_power = False,
                    ),
            nominal_thermal_power = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter='nominal_thermal_power',
                    label='Nominal thermal power [Gw] for {reactor}',
                    pars = uncertaindict(
                        [
                            ( 'DB1', 2.895 ),
                            ( 'DB2', 2.895 ),
                            ( 'LA1', 2.895 ),
                            ( 'LA2', 2.895 ),
                            ( 'LA3', 2.895 ),
                            ( 'LA4', 2.895 ),
                            ],
                        uncertainty = 0.5,
                        mode = 'percent',
                        ),
                    ),
            snf_ff = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'snf_fission_fractions',
                    objectize=True,
                    label='SNF fission fraction {isotope}',
                    pars = uncertaindict([
                        ('U235', 0.563),
                        ('U238', 0.079),
                        ('Pu239', 0.301),
                        ('Pu241', 0.057)],
                        mode = 'fixed',
                        ),
                    ),
            snf_denom = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'snf_denom',
                    objectize=True,
                    label='SNF inverse weight for average energy release per decay',
                    pars = uncertain(snf_energy_per_decay, 'fixed'),
                    ),
            snf_correction = NestedDict(
                bundle = dict(name='reactor_snf_spectra',
                              version='v04', major='r'),
                snf_average_spectra = './data/reactor_anu_spectra/SNF/kopeikin_0412.044_spent_fuel_spectrum_smooth.dat',
                ),
            eper_fission =  NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = "eper_fission",
                    label = 'Energy per fission for {isotope} in MeV',
                    pars = uncertaindict(
                        [('Pu239', (211.12, 0.34)),
                         ('Pu241', (214.26, 0.33)),
                         ('U235',  (202.36, 0.26)),
                         ('U238', (205.99, 0.52))],
                        mode='absolute'
                        ),
                    ),
            conversion_factor =  NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter='conversion_factor',
                    label='Conversion factor from GWt to MeV',
                    #taken from transformations/neutrino/ReactorNorm.cc
                    pars = uncertain(R.NeutrinoUnits.reactorPowerConversion, 'fixed'),
                    ),
            nprotons_nominal =  NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter='nprotons_nominal',
                    label='Daya Bay nominal number of protons (20 tons x GdLS Np/ton)',
                    pars = uncertain(20.0*7.163e28, 'fixed'),
                    ),
            nprotons_corr = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'nprotons_corr',
                    label='Correction to number of protons per AD',
                    pars = uncertaindict([
                        ('AD11', 19941./nominal_mass),
                        ('AD12', 19967./nominal_mass),
                        ('AD21', 19891./nominal_mass),
                        ('AD22', 19944./nominal_mass),
                        ('AD31', 19917./nominal_mass),
                        ('AD32', 19989./nominal_mass),
                        ('AD33', 19892./nominal_mass),
                        ('AD34', 19931./nominal_mass)],
                        mode = 'fixed',
                        ),
                    ),
            iav = NestedDict(
                    bundle     = dict(name='detector_iav_db_root_v03', major='d'),
                    parname    = 'OffdiagScale',
                    scale      = uncertain(1.0, 4, 'percent'),
                    ndiag      = 1,
                    filename   = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
                    matrixname = 'iav_matrix',
                    ),
            eres = NestedDict(
                    bundle = dict(name='detector_eres_normal', version='v01', major=''),
                    # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                    pars = uncertaindict(
                        [('a', 0.016),
                         ('b', 0.081),
                         ('c', 0.026)],
                        #  [('a', 0.014764) ,
                         #  ('b', 0.0869) ,
                         #  ('c', 0.0271)],
                        mode='percent',
                        uncertainty=30
                        ),
                    expose_matrix = True
                    ),
            lsnl = NestedDict(
                    bundle     = dict(name='energy_nonlinearity_db_root', version='v02', major='dl'),
                    names      = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                    filename   = 'data/dayabay/tmp/detector_nl_consModel_450itr.root',
                    par        = uncertain(1.0, 0.2, 'percent'),
                    edges      = 'evis_edges',
                    extrapolation_strategy = 'extrapolate',
                    nonlin_range = (0.5, 12.)
                    ),
            rebin = NestedDict(
                    bundle = dict(name='rebin', version='v03', major=''),
                    rounding = 3,
                    edges = N.concatenate(( [0.7], N.arange(1.2, 8.1, 0.2), [12.0] )),
                    name = 'rebin',
                    label = 'Final histogram\n {detector}'
                    ),
            #
            # Spectra
            #
            bkg_spectrum_acc = NestedDict(
                bundle    = dict(name='root_histograms_v03'),
                filename  = 'data/dayabay/data_spectra/P15A_IHEP_data/P15A_All_raw_sepctrum_coarse.root',
                format    = '{site}_AD{adnum_local}_singleTrigEnergy',
                name      = 'bkg_spectrum_acc',
                label     = 'Accidentals {detector}\n (norm spectrum)',
                groups    = self.groups,
                normalize = True,
                ),
            bkg_spectrum_li=NestedDict(
                bundle    = dict(name='root_histograms_v03'),
                filename  = 'data/dayabay/bkg/lihe/toyli9spec_BCWmodel_v1.root',
                format    = 'h_eVisAllSmeared',
                name      = 'bkg_spectrum_li',
                label     = '9Li spectrum\n (norm)',
                normalize = True,
                ),
            bkg_spectrum_he= NestedDict(
                bundle    = dict(name='root_histograms_v03'),
                filename  = 'data/dayabay/bkg/lihe/toyhe8spec_BCWmodel_v1.root',
                format    = 'h_eVisAllSmeared',
                name      = 'bkg_spectrum_he',
                label     = '8He spectrum\n (norm)',
                normalize = True,
                ),
            bkg_spectrum_amc = NestedDict(
                bundle    = dict(name='root_histograms_v03'),
                filename  = 'data/dayabay/bkg/P12B_amc_expofit.root',
                format    = 'hCorrAmCPromptSpec',
                name      = 'bkg_spectrum_amc',
                label     = 'AmC spectrum\n (norm)',
                normalize = True,
                ),
            bkg_spectrum_alphan = NestedDict(
                bundle    = dict(name='root_histograms_v03'),
                filename  = 'data/dayabay/bkg/P12B_alphan_coarse.root',
                format    = 'AD{adnum_global_alphan_subst}',
                groups    = self.groups,
                name      = 'bkg_spectrum_alphan',
                label     = 'C(alpha,n) spectrum\n {detector} (norm)',
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
            bkg_spectrum_fastn=NestedDict(
                    bundle=dict(name='dayabay_fastn_power', version='v02', major='s'),
                    parameter='fastn_shape',
                    name='bkg_spectrum_fastn',
                    normalize=(0.7, 12.0),
                    bins='evis_edges',
                    order=2,
                    ),
            #
            # Parameters
            #
            fastn_shape=NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter='fastn_shape',
                    label='Fast neutron shape parameter for {site}',
                    pars=uncertaindict(
                        [ ('EH1', (67.79, 0.1132)),
                          ('EH2', (58.30, 0.0817)),
                          ('EH3', (68.02, 0.0997)) ],
                        mode='relative',
                        ),
                    ),
           lihe_fracs = NestedDict(
                    bundle = dict(name='dyb_lihe_fractions',
                                  version='v01'),
                    frac_file_path = './data/dayabay/fractions/{site}_fractions.txt',
                    mode=self.opts.lihe_fractions
                ) if self.opts.lihe_fractions != 'no' else
                 NestedDict(),
            #
            # Rates
            #
            bkg_rate_acc = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'bkg_rate_acc',
                    label='Acc rate at {detector}',
                    pars = uncertaindict(
                        [
                            ( 'AD11', 8.463602 ),
                            ( 'AD12', 8.464611 ),
                            ( 'AD21', 6.290756 ),
                            ( 'AD22', 6.180896 ),
                            ( 'AD31', 1.273798 ),
                            ( 'AD32', 1.189542 ),
                            ( 'AD33', 1.197807 ),
                            ( 'AD34', 0.983096 ),
                            ],
                        uncertainty = 1.0,
                        mode = 'percent',
                        ),
                    separate_uncertainty='acc_norm',
                    ),
            bkg_rate_lihe = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'bkg_rate_lihe',
                    label='⁹Li/⁸He rate at {site}',
                    pars = uncertaindict([
                        ('EH1', (2.46, 1.06)),
                        ('EH2', (1.72, 0.77)),
                        ('EH3', (0.15, 0.06))],
                        mode = 'absolute',
                        ),
                    ),
            bkg_rate_fastn = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'bkg_rate_fastn',
                    label='Fast neutron rate at {site}',
                    pars = uncertaindict([
                        ('EH1', (0.792, 0.103)),
                        ('EH2', (0.566, 0.074)),
                        ('EH3', (0.047, 0.009))],
                        mode = 'absolute',
                        ),
                    ),
            bkg_rate_amc = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'bkg_rate_amc',
                    label='AmC rate at {detector}',
                    pars = uncertaindict(
                        [
                            ('AD11', (0.18, 0.08)),
                            ('AD12', (0.18, 0.08)),
                            ('AD21', (0.16, 0.07)),
                            ('AD22', (0.15, 0.07)),
                            ('AD31', (0.07, 0.03)),
                            ('AD32', (0.06, 0.03)),
                            ('AD33', (0.07, 0.03)),
                            ('AD34', (0.05, 0.02)),
                            ],
                        mode = 'absolute',
                        ),
                    ),
            bkg_rate_alphan = NestedDict(
                    bundle = dict(name="parameters", version = "v01"),
                    parameter = 'bkg_rate_alphan',
                    label='C(alpha,n) rate at {detector}',
                    pars = uncertaindict([
                            ('AD11', (0.08)),
                            ('AD12', (0.07)),
                            ('AD21', (0.05)),
                            ('AD22', (0.07)),
                            ('AD31', (0.05)),
                            ('AD32', (0.05)),
                            ('AD33', (0.05)),
                            ('AD34', (0.05)),
                            ],
                            uncertainty = 50,
                            mode = 'percent',
                        ),
                    ),
        )

    def build(self):
        # Initialize the expression and indices
        l = list(chain.from_iterable(self.formula.values()))
        self.expression = Expression_v01(l, self.nidx)

        # Dump the information
        if self.opts.verbose:
            print(self.expression.expressions_raw)
            print(self.expression.expressions)

        # Parse the expression
        self.expression.parse()
        # The next step is needed to name all the intermediate variables.
        lib = self.libs
        self.expression.guessname(lib, save=True)

        if self.opts.verbose and self.opts.verbose>1:
            print('Expression tree:')
            self.expression.tree.dump(True)
            print()

        # Put the expression into context
        self.context = ExpressionContext_v01(self.cfg, ns=self.namespace)
        self.expression.build(self.context)
        self.correlate_escale_and_eff()

        if self.opts.verbose and self.opts.verbose>1:
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

    def correlate_escale_and_eff(self):
        #TODO: fix with proper bundle
        root = self.namespace
        escale_ns = root('escale')
        eff_ns = root('effunc_uncorr')
        for ad in escale_ns.keys():
            rel_escale = escale_ns[ad]
            rel_eff = eff_ns[ad]
            sigma_escale, sigma_eff = rel_escale.sigma(), rel_eff.sigma()
            cov = sigma_escale*sigma_eff * 0.545
            rel_eff.setCovariance(rel_escale, cov)

    def register(self):
        ns = self.namespace
        outputs = self.context.outputs
        reactors  = self.nidx.indices['r'].variants
        isotopes  = self.nidx.indices['i'].variants
        detectors = self.nidx.indices['d'].variants
        components = self.nidx.indices['c'].variants
        time_windows = self.nidx.indices['w'].variants
        if self.opts.embed:
            import IPython
            IPython.embed()

        ns.addobservable("ibd_xsec", outputs.ibd_xsec, export=False)
        ns.addobservable("Enu", outputs.enu, export=False)
        ns.addobservable("Evis", outputs.evis, export=False)
        ns.addobservable("Evis_edges", outputs.evis_edges, export=False)
        ns.addobservable("Evis_centers", outputs.evis_centers, export=False)
        ns.addobservable("ctheta", outputs.ctheta, export=False)
        ns.addobservable("ee", outputs.ee, export=False)
        ns.addobservable("jacobian", outputs.jacobian, export=False)
        ns.addobservable("iav_matrix_raw", outputs.iavmatrix_raw)

        for iso in isotopes:
            ns.addobservable("anuespec.{0}".format(iso), outputs.anuspec[iso], export=False)

        for reac in reactors:
            ns.addobservable("thermal_power.{0}".format(reac), outputs.thermal_power[reac], export=False)

        for ad in self.detectors:
            if self.opts.lihe_fractions != 'no':
                for w in time_windows:
                    ns.addobservable("bkg.{0}.{1}".format(ad, w), outputs.bkg[ad][w], export=False)
            else:
                ns.addobservable("bkg.{0}".format(ad),         outputs.bkg[ad], export=False)
            ns.addobservable("bkg.acc.{0}".format(ad),         outputs.bkg_acc[ad], export=False)
            ns.addobservable("bkg.fastn.{0}".format(ad), outputs.bkg_fastn[ad], export=False)
            ns.addobservable("bkg.amc.{0}".format(ad), outputs.bkg_amc[ad], export=False)
            ns.addobservable("bkg.alphan.{0}".format(ad), outputs.bkg_alphan[ad], export=False)
            ns.addobservable("bkg.lihe.{0}".format(ad), outputs.bkg_lihe[ad], export=False)

            for reac in reactors:
                ns.addobservable('oscprob.{0}.{1}'.format(ad, reac), outputs.osc_prob_rd[ad][reac])

            else:
                ns.addobservable("reactor_pred.{0}".format(ad),
                        outputs.integral[ad], export=False)

            ns.addobservable("iav_matrix.{0}".format(ad), outputs.iavmatrix[ad])
            ns.addobservable("iav.{0}".format(ad), outputs.iav[ad])
            ns.addobservable("lsnl.{0}".format(ad), outputs.lsnl[ad])
            ns.addobservable("eres.{}".format(ad), outputs.eres[ad])
            if self.opts.lihe_fractions != 'no':
                for w in time_windows:
                    ns.addobservable("{0}.{1}".format(ad, w), outputs.rebin[ad][w])
            else:
                ns.addobservable("{0}".format(ad), outputs.rebin[ad])
            ns.addobservable("final_concat", outputs.total)

    def print_stats(self):
        from gna.graph import GraphWalker, report, taint, taint_dummy
        out=self.context.outputs.concat_total
        walker = GraphWalker(out)
        report(out.data, fmt='Initial execution time: {total} s')
        report(out.data, 100, pre=lambda: walker.entry_do(taint), pre_dummy=lambda: walker.entry_do(taint_dummy))
        print('Statistics', walker.get_stats())
        print('Parameter statistics', self.stats)

    def define_topology(self):
        self.formula['base'] = [
            # Basic building blocks
            'baseline[d,r]',
            'enu| ee(evis()), ctheta()',
            'fissions = fission_fractions[i,r]()',
            'eper_ff_var = eper_fission[i]*fission_fraction_corr[i,r]',
            'ff = fission_fraction_corr * fissions',
            'denom = sum[i] |eper_ff_var*fissions',
            #  'inv_denom = inverse()| denom',
            'nprotons_ad = nprotons_nominal*nprotons_corr[d]',
            'anuspec[i](enu())',
        ]
        self.formula['livetime'] = [
            'efflivetime=accumulate("efflivetime", efflivetime_daily[d]())',
            'livetime=accumulate("livetime", livetime_daily[d]())',
            'numenator = efflivetime_daily[d]()*nominal_thermal_power[r]*thermal_power[r]()*ff ',
            'power_livetime_factor_daily = numenator / denom',
            'power_livetime_factor=accumulate("power_livetime_factor", power_livetime_factor_daily)',
            ]

        if self.opts.no_snf:
            pass
        else:
            self.formula['snf'] = [
                    'snf_plf_daily = nominal_thermal_power[r]*snf_fission_fractions[i]() * snf_denom',
                    'nominal_spec_per_reac =  sum[i]| snf_plf_daily*anuspec[i](enu())',
                    'snf_in_reac = snf_correction(enu(), nominal_spec_per_reac)',
                    ]


        # Bkg
        self.formula['bkg'] = [
            'fastn_shape[s]',
            'bkg_acc      = days_in_second * efflivetime[d] * bkg_rate_acc[d]    * acc_norm[d] * bkg_spectrum_acc[d]()',
            'bkg_fastn    = days_in_second * efflivetime[d] * bkg_rate_fastn[s]                * bkg_spectrum_fastn[s]()',
            'bkg_amc      = days_in_second * efflivetime[d] * bkg_rate_amc[d]                  * bkg_spectrum_amc()',
            'bkg_alphan   = days_in_second * efflivetime[d] * bkg_rate_alphan[d]               * bkg_spectrum_alphan[d]()',
            'bkg_lihe     = days_in_second * efflivetime[d] * bkg_rate_lihe[s]   * bracket| frac_li * bkg_spectrum_li() + frac_he * bkg_spectrum_he()',
            'norm_bf = global_norm*eff*effunc_uncorr[d]'
            ]
        if self.opts.lihe_fractions != 'no':
            self.formula['bkg_total'] = [
                    'bkg = bracket|  lihe_frac[s,w]*bkg_lihe + ibd_frac[s,w]*bracket| bkg_acc + bkg_fastn + bkg_amc + bkg_alphan',
                    ]
        else:
            self.formula['bkg_total'] = [
                    'bkg = bracket|  bkg_lihe +  bkg_acc + bkg_fastn + bkg_amc + bkg_alphan',
                    ]

        if self.opts.no_snf:
            self.formula['anue_spectra'] = [
                '''anue_rd = ibd_xsec(enu(), ctheta())*jacobian(enu(), ee(), ctheta())*
                                    (sum[i]| power_livetime_factor*offeq_correction[i,r](enu(), anuspec[i]()))
                ''']
        else:
            self.formula['anue_spectra'] = [
                '''anue_rd = ibd_xsec(enu(), ctheta())*jacobian(enu(), ee(), ctheta())*
                                    ((sum[i]|
                                    power_livetime_factor*offeq_correction[i,r](enu(),
                                    anuspec[i]())) + efflivetime*snf_in_reac)
                ''']

        self.formula['oscprob'] = [ '''osc_prob_rd = sum[c]| pmns[c]*oscprob[c,d,r](enu())''' ]

        # Some glue
        self.formula['glue'] = [
            '''unoscillated_reactor_flux_in_det = conversion_factor*nprotons_ad*baselineweight[r,d]*anue_rd
            ''',
            '''oscillated_spectra_d = sum[r]| osc_prob_rd*unoscillated_reactor_flux_in_det''',
            '''oscillated_spectra_in_det = integral| norm_bf* oscillated_spectra_d''',
        ]

        # Detector effects initialization
        self.formula['det_effects_base'] = [
            'eres_matrix| evis_hist()',
            'lsnl_edges| evis_hist(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()'
            ]

        self.formula['prediction'] = ['''ibd =
                          eres[d]|
                            lsnl[d]|
                              iav[d]|
                                  oscillated_spectra_in_det''']

        self.formula['noeffects'] = [
                'observation_noeffects=norm_bf*conversion_factor*nprotons_nominal*eres()'
                ]

        if self.opts.lihe_fractions != 'no':
             self.formula['observation'] = [
                     'observation=rebin| ibd_frac[s,w]*ibd + bkg',
                     'total=concat[w]| concat[d]| observation'
                ]
        else:
             self.formula['observation'] = [
                     'observation=rebin| ibd + bkg',
                     'concat_total=concat[d]| observation'
                ]


    def define_labels(self):
        self.libs =  dict(
                      eper_ff_var  = dict(expr='eper_fission[i]*fission_fraction_corr[i,r]',label='Product of energy per fission to fission fraction'),
                        eff_corrected_unosc_spectra = dict(expr=('norm_bf * unoscillated_spectra_d', 'eff_corrected_unosc_spectra'),
                                                           label='Eff corrected unosc spectra'),
                        unoscillated_spectra_d = dict(expr=('unoscillated_spectra_d', 'sum[r]| unoscillated_reactor_flux_in_det'),
                                                           label='Unoscillated flux in {detector}, not integrated'),
                        iav = dict(expr=('iav[d]| unoscillated_spectra_in_det', 'iav[d]| oscillated_spectra_in_det', 'iav[d]| integral'),
                                   label='Anue spectra in {detector after IAV}'),
                        anue_rd = dict(expr='anue_rd',
                                       label='Anue in {detector} from {reactor}'),
                        osc_prob_rd = dict(expr='osc_prob_rd',
                                           label='Oscillation probability from {reactor} to {detector}'),
                        osc_rate = dict(expr='anue_rd*osc_prob_rd',
                                        label='Oscillated spectra from {reactor} in {detector}'),
                        osc_pred_det = dict(expr='integral| sum[r]| baselineweight*conversion_factor*nprotons_nominal*osc_rate',
                                            label='Oscillated prediction in {detector}'),
                        ibd_cross_section = dict(expr='ibd_xsec*jacobian',
                                                 label='IBD cross section in first order'),
                        osccomp_spectra_in_det = dict(expr='osccomp_spectra_in_det',
                                                            label='Integrated reactor spectra * by osc comp+PMNS weight {component} in {detector}'),
                        oscillated_spectra_in_det = dict(expr='integral| oscillated_spectra_d',
                                                        label='Oscillated reactor spectra in {detector}'),
                        unoscillated_reactor_flux_in_det = dict(expr='unoscillated_reactor_flux_in_det',
                                                             label='Unoscillated spectra in {detector}'),
                        #  unoscillated_reactor_spectra = dict(expr='unoscillated_reactor_spectra',
                                                             #  label='Unoscillated spectra in {detector}'),
                        rate_in_detector        = dict(expr='rate_in_detector',
                                                       label='Rate in {detector}' ),
                        #  denom                   = dict(expr='denom',
                                                       #  label='Denominator of reactor norm'),
                        anue_produced_iso       = dict(expr='power_livetime_factor*anuspec',
                                                       label='Total number of anue produced for {isotope} in {reactor}@{detector}'),
                        anue_produced_total     = dict(expr='sum:i|anue_produced_iso',
                                                       label='Total number of anue produced in {reactor}@{detector}'),
                        xsec_weighted           = dict(expr='baselineweight*ibd_xsec',
                                                       label="Cross section weighted by distance {reactor}@{detector}"),
                        count_rate_rd           = dict(expr='anue_produced_total*jacobian*xsec_weighted',
                                                       label='Countrate from {reactor} in {detector}'),
                        count_rate_d            = dict(expr='sum:r|countrate_rd',
                                                       label='Countrate in {detector}'),
                        numenator               = dict(expr='efflivetime_daily*thermal_power*ff',
                                                       label='Numenator of normalization  for {reactor}@{detector} ({isotope})'),

                        cspec_diff              = dict(expr='anuspec*ibd_xsec*jacobian*oscprob',
                                                       label='anu count rate\n {isotope}@{reactor}->{detector} ({component})'),
                        # cspec_diff_reac         = dict(expr='sum:i'),
                        cspec_diff_reac_l       = dict(expr='baselineweight*cspec_diff_reac'),
                        # cspec_diff_det          = dict(expr='sum:r'),
                        # spec_diff_det           = dict(expr='sum:c'),
                        cspec_diff_det_weighted = dict(expr='pmns*cspec_diff_det'),

                        norm_bf                 = dict(expr='eff*effunc_uncorr*global_norm'),
                        prediction_scale        = dict(expr='conversion_factor*norm_bf*nprotons_nominal'),
                        ibd                     = dict(expr='eres*norm_bf', label='Observed IBD spectrum \n {detector}'),


                        lsnl_component_weighted = dict(expr='lsnl_component*lsnl_weight',
                                                       label='Weighted LSNL curve {lsnl_component}'),
                        lsnl_correlated         = dict(expr='sum:l|lsnl_component_weighted',
                                                       label='Sum of LSNL curves'),
                        evis_after_escale       = dict(expr='escale*evis_edges',
                                                       label='Evis edges after escale, {detector}'),
                        evis_nonlinear_correlated = dict(expr='evis_edges*lsnl_correlated'),
                        #  evis_nonlinear_correlated = dict(expr='evis_after_escale*lsnl_correlated',
                        evis_nonlinear          = dict(expr='escale*evis_nonlinear_correlated',
                                                       label='Edges after LSNL, {detector}'),
                        oscprob_weighted        = dict(expr='oscprob*pmns'),
                        oscprob_full            = dict(expr='sum:c|oscprob_weighted',
                                                       label='anue survival probability\n {reactor}@{detector}'),
                        #  oscillation_probability = dict(expr='sum:d|oscprob_full',
                                                       #  label='anue survival probability\n {reactor}@{detector}'),

                        anuspec_weighted        = dict(expr='anuspec*power_livetime_factor'),
                        anuspec_rd              = dict(expr='sum:i|anuspec_weighted', label='anue spectrum {reactor}->{detector}\n weight: {weight_label}'),

                        countrate_rd            = dict(expr='anue_produced_total*ibd_xsec*jacobian*osc_prob_rd',
                                                       label='Countrate {reactor}@{detector}'),
                        countrate_weighted      = dict(expr='baselineweight*countrate_rd'),
                        countrate               = dict(expr='sum:r|countrate_weighted', label='Count rate {detector}\n weight: {weight_label}'),

                        observation_fine        = dict(expr='bkg+ibd', label='Observed spectrum \n {detector}'),

                        # Accidentals
                        acc_num_bf        = dict(expr='acc_norm*bkg_rate_acc*days_in_second*efflivetime',             label='Acc num {detector}\n (best fit)}'),
                        bkg_acc           = dict(expr='acc_num_bf*bkg_spectrum_acc',                   label='Acc {detector}\n (w: {weight_label})'),

                        # Li/He
                        bkg_spectrum_li_w = dict(expr='bkg_spectrum_li*frac_li',                       label='9Li spectrum\n (frac)'),
                        bkg_spectrum_he_w = dict(expr='bkg_spectrum_he*frac_he',                       label='8He spectrum\n (frac)'),
                        bkg_spectrum_lihe = dict(expr='bkg_spectrum_he_w+bkg_spectrum_li_w',           label='8He/9Li spectrum\n (norm)'),
                        lihe_num_bf       = dict(expr='bkg_rate_lihe*days_in_second*efflivetime'),
                        bkg_lihe          = dict(expr='bkg_spectrum_lihe*lihe_num_bf',                 label='8He/9Li {detector}\n (w: {weight_label})'),

                        # Fast neutrons
                        fastn_num_bf      = dict(expr='bkg_rate_fastn*days_in_second*efflivetime'),
                        bkg_fastn         = dict(expr='bkg_spectrum_fastn*fastn_num_bf',               label='Fast neutron {detector} (w: {weight_label})'),

                        # AmC
                        amc_num_bf        = dict(expr='bkg_rate_amc*days_in_second*efflivetime'),
                        bkg_amc           = dict(expr='bkg_spectrum_amc*amc_num_bf',                   label='AmC {detector}\n (w: {weight_label})'),

                        # AlphaN
                        alphan_num_bf     = dict(expr='bkg_rate_alphan*days_in_second*efflivetime'),
                        bkg_alphan        = dict(expr='bkg_spectrum_alphan*alphan_num_bf',             label='C(alpha,n) {detector}\n (w: {weight_label})'),

                        # Total background
                        bkg               = dict(expr='bkg_acc+bkg_alphan+bkg_amc+bkg_fastn+bkg_lihe', label='Background spectrum\n {detector}'),

                        # dybOscar mode
                        eres_cw           = dict(expr='eres*pmns'),
                    )
