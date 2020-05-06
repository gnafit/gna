# reimplementation of ../bundles_legacy/detector_nonlinearity_db_root_v02

# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from gna.env import env, namespace
from gna.configurator import NestedDict
from collections import OrderedDict
from gna.bundle import TransformationBundle
from gna.context import entryContext

class energy_nonlinearity_birks_cherenkov_v01(TransformationBundle):
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.storage=NestedDict()

    @staticmethod
    def _provides(cfg):
        return (), ('evis_edges_hist', 'lsnl')

    def init_data(self):
        dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]
        self.stopping_power=N.loadtxt(self.cfg.stopping_power, dtype=dtype_spower)

        from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
        cfg = self.cfg.annihilation_electrons
        file = R.TFile(cfg.file, 'read')
        hist = file.Get(cfg.histogram)
        buf=get_buffer_hist1(hist).copy()
        buf*=cfg.scale
        edges = get_bin_edges_axis(hist.GetXaxis())

        self.annihilation_electrons_p_input = buf
        self.annihilation_electrons_edges_input = edges
        self.annihilation_electrons_centers_input = 0.5*(edges[1:]+edges[:-1])

        file.Close()

    def build(self):
        with entryContext(subgraph='LSNL'):
            self.init_data()

            #
            # Initialize bin edges
            #
            self.histoffset = C.HistEdgesOffset(self.doubleme, labels='Offset/threshold bin edges (Evis, Te+)')
            histedges = self.histoffset.histedges

            # Declare open input
            self.set_input('evis_edges_hist', None, histedges.hist_in, argument_number=0)

            histedges.points.setLabel('Evis (full range)')
            histedges.points_truncated.setLabel('Evis (truncated)')
            histedges.points_threshold.setLabel('Evis (threshold)')
            histedges.points_offset.setLabel('Te+')
            histedges.hist_truncated.setLabel('Hist Evis (truncated)')
            histedges.hist_threshold.setLabel('Hist Evis (threshold)')
            histedges.hist_offset.setLabel('Hist Te+')

            #
            # Birk's model integration
            #
            birks_e_input, birks_quenching_input = self.stopping_power['e'], self.stopping_power['dedx']
            self.birks_e_p, self.birks_quenching_p = C.Points(birks_e_input, labels='Te (input)'), C.Points(birks_quenching_input, labels='Stopping power (dE/dx)')

            birksns = self.namespace('birks')
            with birksns:
                self.birks_integrand_raw = C.PolyRatio([], list(sorted(birksns.storage.keys())), labels="Birk's integrand")
            self.birks_quenching_p >> self.birks_integrand_raw.polyratio.points

            self.doubleemass_point = C.Points([-self.doubleme], labels='2me offset')

            self.integrator_ekin = C.IntegratorGL(self.histoffset.histedges.hist_offset, self.cfg.integration_order, labels=('Te sampler (GL)', "Birk's integrator (GL)"))

            self.birks_integrand_interpolator = C.InterpLogx(self.birks_e_p, self.integrator_ekin.points.x, labels=("Birk's InSegment", "Birk's interpolator"))
            self.birks_integrand_interpolated = self.birks_integrand_interpolator.add_input(self.birks_integrand_raw.polyratio.ratio)
            self.birks_integral = self.integrator_ekin.add_input(self.birks_integrand_interpolated)

            self.birks_accumulator = C.PartialSum(0., labels="Birk's Evis|[MeV]")
            self.birks_integral >> self.birks_accumulator.reduction

            #
            # Cherenkov model
            #
            with self.namespace('cherenkov'):
                self.cherenkov = C.Cherenkov_Borexino(labels='Npe Cherenkov')
            self.histoffset.histedges.points_offset >> self.cherenkov.cherenkov

            #
            # Electron energy model
            #
            with self.namespace:
                self.electron_model = C.WeightedSum(['kC', 'Npescint'], [self.cherenkov.cherenkov.ch_npe, self.birks_accumulator.reduction.out], labels='Npe: electron responce')

            #
            # 2 511 keV gamma model
            #
            self.annihilation_electrons_centers = C.Points(self.annihilation_electrons_centers_input, labels='Annihilation gamma E centers')
            self.annihilation_electrons_p = C.Points(self.annihilation_electrons_p_input, labels='Annihilation gamma weights')

            self.view_lowe = C.ViewHistBased(self.histoffset.histedges.hist_offset, 0.0, self.annihilation_electrons_edges_input[-1], labels=('Low E indices', 'Low E view'))
            self.ekin_edges_lowe = self.view_lowe.add_input(self.histoffset.histedges.points_offset)
            self.electron_model_lowe = self.view_lowe.add_input(self.electron_model.single())

            self.ekin_edges_lowe.setLabel('Te+ edges (low Te view)')
            self.electron_model_lowe.setLabel('Npe: electron responce (low Te view)')

            self.electron_model_lowe_interpolator = C.InterpLinear(self.ekin_edges_lowe, self.annihilation_electrons_centers, labels=('Annihilation E InSegment', 'Annihilation gamma interpolator'))
            self.electron_model_lowe_interpolated = self.electron_model_lowe_interpolator.add_input(self.electron_model_lowe)

            with self.namespace:
                self.npe_positron_offset = C.Convolution('ngamma', labels='e+e- annihilation Evis [MeV]')
                self.electron_model_lowe_interpolated >> self.npe_positron_offset.normconvolution.fcn
                self.annihilation_electrons_p >> self.npe_positron_offset.normconvolution.weights

            #
            # Total positron model
            #
            self.positron_model = C.SumBroadcast(outputs=[
                self.electron_model.sum.sum,
                self.npe_positron_offset.normconvolution.result
                ],
                labels='Npe: positron responce')

            self.positron_model_scaled = C.FixedPointScale(self.histoffset.histedges.points_truncated, self.namespace['normalizationEnergy'], labels=('Fixed point index', 'Positron energy model|Evis, MeV'))
            self.positron_model_scaled = self.positron_model_scaled.add_input(self.positron_model.sum.outputs[0])
            self.positron_model_scaled_full_view = C.ViewRear(-1.0, labels='Positron Energy nonlinearity|full range')
            self.positron_model_scaled_full_view.determineOffset(self.histoffset.histedges.hist, self.histoffset.histedges.hist_truncated, True)
            self.histoffset.histedges.points >> self.positron_model_scaled_full_view.view.original
            self.positron_model_scaled >> self.positron_model_scaled_full_view.view.rear
            self.positron_model_scaled_full = self.positron_model_scaled_full_view.view.result

            #
            # Relative positron model
            #
            self.positron_model_relative = C.Ratio(self.positron_model_scaled, self.histoffset.histedges.points_truncated, labels='Positron energy nonlinearity')
            self.positron_model_relative_full_view = C.ViewRear(0.0, labels='Positron Energy nonlinearity|full range')
            self.positron_model_relative_full_view.determineOffset(self.histoffset.histedges.hist, self.histoffset.histedges.hist_truncated, True)
            self.histoffset.histedges.points >> self.positron_model_relative_full_view.view.original
            self.positron_model_relative >> self.positron_model_relative_full_view.view.rear
            self.positron_model_relative_full = self.positron_model_relative_full_view.view.result

        #
        # Hist Smear
        #
        self.pm_histsmear = C.HistNonlinearity(self.cfg.get('fill_matrix', False), labels=('Nonlinearity matrix', 'Nonlinearity smearing'))
        self.pm_histsmear.set_range(-0.5, 20.0)
        self.positron_model_scaled_full >> self.pm_histsmear.matrix.EdgesModified

        self.histoffset.histedges.hist >> self.pm_histsmear.matrix.Edges

        trans = self.pm_histsmear.transformations.back()
        for i, it in enumerate(self.nidx.iterate()):
            # if i:
                # trans = self.pm_histsmear.add_transformation()
            inp = self.pm_histsmear.add_input()

            trans.setLabel(it.current_format('Nonlinearity smearing {autoindex}'))

            self.set_input('lsnl', it, inp, argument_number=0)
            self.set_output('lsnl', it, trans.outputs.back())

    def define_variables(self):
        with entryContext(subgraph="LSNL"):
            from physlib import pdg
            ns = self.namespace

            #
            # Some constants
            #
            emass = ns.reqparameter("emass", central=pdg['live']['ElectronMass'], fixed=True, label='Electron mass, MeV')
            self.doubleme = 2*emass.value()
            ns.defparameter("ngamma", central=2.0, fixed=True, label='Number of e+e- annihilation gammas')

            labels=OrderedDict([
                ( 'birks.Kb0', 'Kb0=1' ),
                ( 'birks.Kb1', "Birk's 1st constant (E')" ),
                ( 'birks.Kb2', "Birk's 2nd constant (E'')" ),
                ( 'cherenkov.E_0', '' ),
                ( 'cherenkov.p0', '' ),
                ( 'cherenkov.p1', '' ),
                ( 'cherenkov.p2', '' ),
                ( 'cherenkov.p3', '' ),
                ( 'cherenkov.p4', '' ),
                ( 'Npescint', '' ),
                ( 'kC', '' ),
                ( 'normalizationEnergy', '' )
                ])

            #
            # Define parameters according to predefined list and input configuration
            #
            for name, label in labels.items():
                parcfg = self.cfg.pars.get(name, None)
                if parcfg is None:
                    if 'Kb2' in name:
                        continue
                    raise self.exception('Parameter {} configuration is not provided'.format(name))
                self.reqparameter(name, None, cfg=parcfg, label=label)

            #
            # Set the correlation matrix if provided by configuration
            #
            correlations_pars = self.cfg.correlations_pars
            correlations = self.cfg.correlations

            if not correlations_pars and not correlations:
                return
            if not correlations_pars or not correlations:
                raise self.exception("Both 'correlations' and 'correlations_pars' should be defined")

            npars = len(correlations_pars)
            corrmatrix = N.array(correlations, dtype='d').reshape(npars, npars)

            # Check matrix sanity
            if (corrmatrix.diagonal()!=1.0).any():
                raise self.exception('There should be no non-unitary elements on the correlation matrix')
            if (N.fabs(corrmatrix)>1.0).any():
                raise self.exception('Correlation matrix values should be within -1.0 and 1.0')
            if (corrmatrix!=corrmatrix.T).all():
                raise self.exception('Correlation matrix is expected to be diagonal')

            # Take parameters and set correlations
            from gna.parameters import covariance_helpers as ch
            pars=[ns[par] for par in correlations_pars]
            ch.covariate_pars(pars, corrmatrix)

