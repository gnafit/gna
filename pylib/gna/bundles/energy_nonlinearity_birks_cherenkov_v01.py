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

class energy_nonlinearity_birks_cherenkov_v01(TransformationBundle):
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.storage=NestedDict()

    @staticmethod
    def _provides(cfg):
        return (), ('lsnl_edges', 'lsnl')

    def init_data(self):
        dtype_spower = [ ('e', 'd'), ('temp1', 'd'), ('temp2', 'd'), ('dedx', 'd') ]
        self.stopping_power=N.loadtxt(self.cfg.stopping_power, dtype=dtype_spower)

        from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
        filename, objname = self.cfg.annihilation_electrons
        file = R.TFile(filename, 'read')
        hist = file.Get(objname)
        buf = get_buffer_hist1(hist).copy()
        buf/=buf.sum()
        edges = get_bin_edges_axis(hist.GetXaxis())

        self.annihilation_electrons_p_input = buf
        self.annihilation_electrons_edges_input = edges
        self.annihilation_electrons_centers_input = 0.5*(edges[1:]+edges[:-1])

    def build(self):
        self.init_data()
        #
        # Birk's model integration
        #
        binwidth=0.025

        self.epos_edges_full_input = N.arange(0.0, 12.0+1.e-6, binwidth)
        self.epos_firstbin = N.where(self.epos_edges_full_input>self.doubleme)[0][0]-1
        self.epos_edges_input=self.epos_edges_full_input[self.epos_firstbin:]
        self.epos_edges_threshold_input = self.epos_edges_input.copy()
        self.epos_edges_threshold_input[0]=self.doubleme

        self.integrator_epos = C.IntegratorGL(self.epos_edges_threshold_input, 3, labels=('Evis sampler (GL)', 'Evis integrator (GL)'))
        self.epos_edges_threshold = self.integrator_epos.points.xedges
        self.epos_edges = C.Points(self.epos_edges_input, labels='Epos edges (no threshold)')
        self.epos_edges_full = C.Points(self.epos_edges_full_input, labels='Epos edges (full range)')

        birks_e_input, birks_quenching_input = self.stopping_power['e'], self.stopping_power['dedx']
        self.birks_e_p, self.birks_quenching_p = C.Points(birks_e_input, labels='Te (input)'), C.Points(birks_quenching_input, labels='Stopping power (dE/dx)')

        birksns = self.namespace('birks')
        with birksns:
            self.birks_integrand_raw = C.PolyRatio([], list(birksns.storage.keys()), labels="Birk's integrand")
        self.birks_quenching_p >> self.birks_integrand_raw.polyratio.points

        self.doubleemass_point = C.Points([-self.doubleme], labels='2me offset')

        self.ekin_edges_p = C.SumBroadcast([self.epos_edges_threshold, self.doubleemass_point.points.points], labels='Evis to Te')
        self.ekin_edges_h = C.PointsToHist(self.ekin_edges_p, labels='Te bin edges')

        self.integrator_ekin = C.IntegratorGL(self.ekin_edges_h.adapter.hist, 2, labels=(('Te sampler (GL)', "Birk's integrator (GL)")))

        self.birks_integrand_interpolator = C.InterpLinear(self.birks_e_p, self.integrator_ekin.points.x, labels=("Birk's InSegment", "Birk's interpolator"))
        self.birks_integrand_interpolated = self.birks_integrand_interpolator.add_input(self.birks_integrand_raw.polyratio.ratio)
        self.birks_integral = self.integrator_ekin.add_input(self.birks_integrand_interpolated)

        self.birks_accumulator = C.PartialSum(0., labels="Birk's Evis|[MeV]")
        self.birks_integral >> self.birks_accumulator.reduction

        #
        # Cherenkov model
        #
        with self.namespace('cherenkov'):
            self.cherenkov = C.Cherenkov_Borexino(labels='Npe Cherenkov')
        self.ekin_edges_p >> self.cherenkov.cherenkov

        #
        # Electron energy model
        #
        with self.namespace:
            self.electron_model = C.WeightedSum(['kC', 'Npesc'], [self.cherenkov.cherenkov.ch_npe, self.birks_accumulator.reduction.out], labels='Npe: electron responce')

        #
        # 2 511 keV gamma model
        #
        self.annihilation_electrons_centers = C.Points(self.annihilation_electrons_centers_input, labels='Annihilation gamma E centers')
        self.annihilation_electrons_p = C.Points(self.annihilation_electrons_p_input, labels='Annihilation gamma weights')
        egamma_offset = 0
        lastpoint = N.where(self.ekin_edges_p.data()>self.annihilation_electrons_edges_input[-1])[0][0]+1
        self.ekin_edges_lowe = C.View(self.ekin_edges_p, egamma_offset, lastpoint-egamma_offset, labels='Te Low E view')
        self.electron_model_lowe = C.View(self.electron_model.single(), egamma_offset, lastpoint-egamma_offset, labels='{Npe: electron responce|(low E view)}')

        self.electron_model_lowe_interpolator = C.InterpLinear(self.ekin_edges_lowe.view.view, self.annihilation_electrons_centers, labels=('Annihilation E InSegment', 'Annihilation gamma interpolator'))
        self.electron_model_lowe_interpolated = self.electron_model_lowe_interpolator.add_input(self.electron_model_lowe.view.view)

        with self.namespace:
            self.npe_positron_offset = C.NormalizedConvolution('ngamma', labels='e+e- annihilation Evis [MeV]')
            self.electron_model_lowe_interpolated >> self.npe_positron_offset.normconvolution.fcn
            self.annihilation_electrons_p >> self.npe_positron_offset.normconvolution.weights

        #
        # Total positron model
        #
        self.positron_model = C.SumBroadcast([self.electron_model.sum.sum, self.npe_positron_offset.normconvolution.result],
                                             labels='Npe: positron responce')

        self.positron_model_scaled = C.FixedPointScale(self.epos_edges, self.namespace['normalizationEnergy'], labels=('Fixed point index', 'Positron energy model|Evis, MeV'))
        self.positron_model_scaled = self.positron_model_scaled.add_input(self.positron_model.sum.outputs[0])
        self.positron_model_scaled_full = C.ViewRear(self.epos_edges_full, self.epos_firstbin, self.epos_edges_input.size, -1.0, labels='Positron Energy nonlinearity|full range')
        self.positron_model_scaled >> self.positron_model_scaled_full.view.rear

        #
        # Relative positron model
        #
        self.positron_model_relative = C.Ratio(self.positron_model_scaled, self.epos_edges, labels='Positron energy nonlinearity')
        self.positron_model_relative_full = C.ViewRear(self.epos_edges_full, self.epos_firstbin, self.epos_edges_input.size, 0.0, labels='Positron Energy nonlinearity\nfull range')
        self.positron_model_relative >> self.positron_model_relative_full.view.rear

        #
        # Hist Smear
        #
        self.pm_histsmear = C.HistNonlinearity(self.cfg.get('fill_matrix', False), labels=('Nonlinearity matrix', 'Nonlinearity smearing'))
        self.pm_histsmear.set_range(-0.5, 20.0)
        self.positron_model_scaled_full >> self.pm_histsmear.matrix.EdgesModified

        self.set_input('lsnl_edges', None, self.pm_histsmear.matrix.Edges, argument_number=0)

        trans = self.pm_histsmear.transformations.back()
        for i, it in enumerate(self.nidx.iterate()):
            # if i:
                # trans = self.pm_histsmear.add_transformation()
            inp = self.pm_histsmear.add_input()

            trans.setLabel(it.current_format('Nonlinearity smearing\n{autoindex}'))

            self.set_input('lsnl', it, inp, argument_number=0)
            self.set_output('lsnl', it, trans.outputs.back())

    def define_variables(self):
        from physlib import pdg
        ns = self.namespace
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
            ( 'Npesc', '' ),
            ( 'kC', '' ),
            ( 'normalizationEnergy', '' )
            ])

        for name, label in labels.items():
            parcfg = self.cfg.pars.get(name, None)
            if parcfg is None:
                if 'Kb2' in name:
                    continue
                raise self.exception('Parameter {} configuration is not provided'.format(name))
            self.reqparameter(name, None, cfg=parcfg, label=label)

