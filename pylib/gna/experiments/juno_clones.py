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
    def initparser(cls, parser, namespace):
        parser.add_argument( '--dot', help='write graphviz output' )
        parser.add_argument( '-s', '--show', action='store_true', help='show the figure' )
        parser.add_argument( '-o', '--output', help='output figure name' )
        parser.add_argument('--stats', action='store_true', help='show statistics')
        parser.add_argument('-p', '--print', action='append', choices=['outputs', 'inputs'], default=[], help='things to print')
        parser.add_argument('-e', '--embed', action='store_true', help='embed')
        parser.add_argument('-c', '--composition', default='complete', choices=['complete', 'minimal'], help='Set the indices coverage')
        parser.add_argument('-m', '--mode', default='simple', choices=['simple', 'dyboscar', 'mid'], help='Set the topology')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')
        parser.add_argument('-z', '--zones', type=int, help='number of zones to split')

    def __init__(self, namespace, opts):
        baseexp.__init__(self, namespace, opts)

        self.init_nidx()
        self.init_formula()
        self.init_configuration()
        self.build()
        self.register()

    def init_nidx(self):
        self.nidx = [
            ('j', 'juno_clone',  ['juno_nh', 'juno_ih']),
            'name',
            ('d', 'detector',    [self.detectorname]),
            ['r', 'reactor',     ['YJ1', 'YJ2', 'YJ3', 'YJ4', 'YJ5', 'YJ6', 'TS1', 'TS2', 'TS3', 'TS4', 'DYB', 'HZ']],
            ['i', 'isotope',     ['U235', 'U238', 'Pu239', 'Pu241']],
            ('c', 'component',   ['comp0', 'comp12', 'comp13', 'comp23']),
            ('l', 'lsnl_component', ['nominal', 'pull0', 'pull1', 'pull2', 'pull3'])
        ]
        if self.opts.composition=='minimal':
            self.nidx[3][2] = self.nidx[3][2][:1]
            self.nidx[4][2] = self.nidx[4][2][:1]

        if self.opts.zones:
            self.nidx.append( ('z', 'eres_zone', ['zone_%02i'%i for i in range(self.opts.zones)]) )

        self.nidx = NIndex.fromlist(self.nidx)

    def init_formula(self):
        self.formula = self.formula_base

        if self.opts.zones:
            self.formula.insert(0, 'zone_weight[z]')

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
                    bundle   = dict(name='integral_2d1d', version='v02', names=dict(integral='kinint2')),
                    variables = ('evis', 'ctheta'),
                    edges    = np.arange(0.0, 12.001, 0.02),
                    xorders   = 2,
                    yorder   = 3,
                    ),
                ibd_xsec = NestedDict(
                    bundle = dict(name='xsec_ibd', version='v02'),
                    order = 1,
                    ),
                oscprob = NestedDict(
                    bundle = dict(name='oscprob', version='v03', major='rdc'),
                    ),
                anuspec = NestedDict(
                    bundle = dict(name='reactor_anu_spectra', version='v03'),
                    name = 'anuspec',
                    filename = ['data/reactor_anu_spectra/Huber/Huber_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat',
                                'data/reactor_anu_spectra/Mueller/Mueller_smooth_extrap_{isotope}_13MeV0.01MeVbin.dat'],
                    # strategy = dict( underflow='constant', overflow='extrapolate' ),
                    edges = np.concatenate( ( np.arange( 1.8, 8.7, 0.5 ), [ 12.3 ] ) ),
                    ),
                eff = NestedDict(
                    bundle = dict(
                        name='efficiencies',
                        version='v02',
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
                fission_fractions = NestedDict(
                    bundle = dict(name="parameters",
                                  version = "v01",
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
                    ),
                livetime = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "livetime",
                        label = 'Livetime of {detector} in seconds',
                        pars = uncertaindict(
                            [('AD1', (6*365*seconds_per_day, 'fixed'))],
                            ),
                        ),
                efflivetime = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "efflivetime",
                        label = 'Effective livetime of {detector} in seconds',
                        pars = uncertaindict(
                            [('AD1', (6*365*seconds_per_day*0.8, 'fixed'))],
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
                        ),
                target_protons = NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "target_protons",
                        label = 'Number of protons in {detector}',
                        pars = uncertaindict(
                            [('AD1', (1.42e33, 'fixed'))],
                            ),
                        ),
                eper_fission =  NestedDict(
                        bundle = dict(name="parameters", version = "v01"),
                        parameter = "eper_fission",
                        label = 'Energy per fission for {isotope} in MeV',
                        pars = uncertaindict(
                            [('Pu239', (209.99, 0.60)),
                             ('Pu241', (213.60, 0.65)),
                             ('U235',  (201.92, 0.46)),
                             ('U238', (205.52, 0.96))],
                            mode='absolute'
                            ),
                        ),
                eres = NestedDict(
                        bundle = dict(name='detector_eres_normal', version='v01', major='', inactive=False),
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                        pars = uncertaindict([
                            ('a', (0.000, 'fixed')) ,
                            ('b', (0.03, 30, 'percent')) ,
                            ('c', (0.000, 'fixed'))
                            ]),
                        expose_matrix = False
                        ),
                eresz = NestedDict(
                        bundle = dict(name='detector_eres_normal', version='v01', major='z', inactive=True),
                        # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
                        parameter = 'eres',
                        pars = uncertaindict([
                            ('zone_00.a', (0.000, 'fixed')) ,
                            ('zone_00.b', (0.02, 30, 'percent')) ,
                            ('zone_00.c', (0.000, 'fixed')),
                            ('zone_01.a', (0.000, 'fixed')) ,
                            ('zone_01.b', (0.03, 30, 'percent')) ,
                            ('zone_01.c', (0.000, 'fixed')),
                            ('zone_02.a', (0.000, 'fixed')) ,
                            ('zone_02.b', (0.04, 30, 'percent')) ,
                            ('zone_02.c', (0.000, 'fixed')),
                            ]),
                        expose_matrix = False
                        ),
                lsnl = NestedDict(
                        bundle     = dict(name='energy_nonlinearity_db_root', version='v02', major='dl'),
                        names      = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
                        filename   = 'data/dayabay/tmp/detector_nl_consModel_450itr.root',
                        par        = uncertain(1.0, 0.2, 'percent'),
                        edges      = 'evis_edges',
                        ),
                rebin = NestedDict(
                        bundle = dict(name='rebin', version='v03', major=''),
                        rounding = 3,
                        edges = np.concatenate(( [0.7], np.arange(1, 8, 0.02), [12.0] )),
                        name = 'rebin',
                        label = 'Final histogram\n{detector} ({juno_clone})'
                        ),
                zones = NestedDict(
                        bundle = dict(name='sphere_eq_volume_cuts_v01')
                        )
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

        self.namespace('juno_nh.pmns')['Alpha'].set('normal')
        self.namespace('juno_ih.pmns')['Alpha'].set('inverted')

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
            self.namespace.printparameters(labels=True)

    def register(self):
        ns = self.namespace
        outputs = self.context.outputs

        nidx = self.nidx.get_subset('jd')
        for it in nidx:
            itj = it.get_subset('j')
            source = it.current_values(name='observation_noeffects')
            target = itj.current_values(name="{0}_noeffects".format(self.detectorname))
            ns('.'.join(target[:-1])).addobservable(target[-1], outputs[source], export=False)

            source = it.current_values(name='ibd')
            target = itj.current_values(name="{0}_fine".format(self.detectorname))
            ns('.'.join(target[:-1])).addobservable(target[-1], outputs[source], export=False)

            source = it.current_values(name='rebin')
            target = itj.current_values(name="{0}".format(self.detectorname))
            ns('.'.join(target[:-1])).addobservable(target[-1], outputs[source], export=True)

    formula_base = [
            'baseline[d,r]',
            'enu| ee(evis()), ctheta()',
            'livetime[d]',
            'efflivetime[d]',
            'eper_fission[i]',
            'power_livetime_factor =  efflivetime[d] * thermal_power[r] * fission_fractions[r,i]',
            # Detector effects
            'eres_matrix[z]| evis_hist()',
            'lsnl_edges| evis_hist(), escale[d]*evis_edges()*sum[l]| lsnl_weight[l] * lsnl_component[l]()',
            'norm_bf = global_norm* eff* effunc_uncorr[d]'
    ]

    formula_ibd_do = '''ibd =
                     norm_bf[d]*
                     sum[c]|
                       pmns[c,j]*
                       eres|
                         lsnl[d]|
                             sum[r]|
                               baselineweight[r,d]*
                               sum[i]|
                                 power_livetime_factor*
                                 kinint2|
                                   anuspec[i](enu())*
                                   oscprob[c,d,r,j](enu())*
                                   ibd_xsec(enu(), ctheta())*
                                   jacobian(enu(), ee(), ctheta())
            '''

    formula_ibd_mid = '''ibd =
                     norm_bf[d]*
                     eres|
                       lsnl[d]|
                           sum[c]|
                             pmns[c,j]*
                             sum[r]|
                               baselineweight[r,d]*
                               sum[i]|
                                 power_livetime_factor*
                                 kinint2|
                                   anuspec[i](enu())*
                                   oscprob[c,d,r,j](enu())*
                                   ibd_xsec(enu(), ctheta())*
                                   jacobian(enu(), ee(), ctheta())
            '''

    formula_ibd_simple = '''ibd =
                            norm_bf[d]*
                            # sum[z]|
                            # zone_weight[z]*
                            # eres[z]
                                eres|
                                  lsnl[d]|
                                        kinint2|
                                          sum[r]|
                                            baselineweight[r,d]*
                                            ibd_xsec(enu(), ctheta())*
                                            jacobian(enu(), ee(), ctheta())*
                                            (sum[i]|  power_livetime_factor*anuspec[i](enu()))*
                                            sum[c]|
                                              pmns[c,j]*oscprob[c,d,r,j](enu())
            '''

    formula_back = [
            'observation_noeffects = norm_bf*kinint2()',
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

