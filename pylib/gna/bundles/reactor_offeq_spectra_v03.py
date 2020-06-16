# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from gna.env import env

class reactor_offeq_spectra_v03(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')

        self.offeq_raw_spectra = dict()
        self._load_data()

    @staticmethod
    def _provides(cfg):
            return ('offeq_scale', 'offeq_norm'), ('offeq_correction',)

    def _load_data(self):
        """Read raw input spectra"""
        data_template = self.cfg.offeq_data
        for isotope in self.nidx.get_subset('i'):
            iso_name, = isotope.current_values()
            datapath = data_template.format(isotope=iso_name)
            try:
                self.offeq_raw_spectra[iso_name] = np.loadtxt(datapath, unpack=True)
            except IOError:
                # U238 doesn't have offequilibrium correction
                if iso_name == 'U238':
                    pass
                else:
                    raise
            assert len(self.offeq_raw_spectra) != 0, "No data loaded"

    def build(self):
        for idx in self.nidx.iterate():
            if 'isotope' in idx.names()[0]:
                iso, reac = idx.current_values()
            else:
                reac, iso = idx.current_values()
            name = "offeq_correction." + idx.current_format()
            try:
                _offeq_energy, _offeq_spectra = list(map(C.Points, self.offeq_raw_spectra[iso]))
                _offeq_energy.points.setLabel("Original energies for offeq spectrum of {}".format(iso))
            except KeyError:
            # U238 doesn't have offequilibrium correction so just pass 1.
                if iso != 'U238':
                    raise
                ones = C.FillLike(1., labels='Offeq correction to {0} spectrum in {1} reactor'.format(iso, reac))
                self.context.objects[name] = ones
                self.set_input('offeq_correction', idx, ones.single_input(), argument_number=0)
                self.set_output("offeq_correction", idx, ones.single())
                continue

            offeq_spectra = C.InterpLinear(labels='Correction for {} spectra'.format(iso))
            offeq_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            offeq_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

            insegment = offeq_spectra.transformations.front()
            insegment.setLabel("Segments")

            interpolator_trans = offeq_spectra.transformations.back()
            interpolator_trans.setLabel("Interpolated spectral correction for {}".format(iso))

            ones = C.FillLike(1., labels="Nominal spectra for {}".format(iso))
            _offeq_energy >> (insegment.edges, interpolator_trans.x)
            _offeq_spectra >> interpolator_trans.y

            self.set_input('offeq_correction', idx, (insegment.points,
                            interpolator_trans.newx, ones.single_input()), argument_number=0)

            par_name = "offeq_scale"
            self.reqparameter(par_name, idx, central=1., relsigma=0.3,
                    labels="Offequilibrium norm for reactor {1} and iso "
                    "{0}".format(iso, reac))
            self.reqparameter("dummy_scale", idx, central=1,
                    fixed=True, labels="Dummy weight for reactor {1} and iso "
                    "{0} for offeq correction".format(iso, reac))

            outputs = [ones.single(), interpolator_trans.single()]
            weights = ['.'.join(("dummy_scale", idx.current_format())),
                       '.'.join((par_name, idx.current_format()))]

            with self.namespace:
                final_sum = C.WeightedSum(weights, outputs, labels='Offeq correction to '
                            '{0} spectrum in {1} reactor'.format(iso, reac))


            self.context.objects[name] = final_sum
            self.set_output("offeq_correction", idx, final_sum.single())

    def define_variables(self):
        pass
