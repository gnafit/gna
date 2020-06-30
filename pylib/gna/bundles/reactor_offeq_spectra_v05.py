# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from gna.env import env

class reactor_offeq_spectra_v05(TransformationBundle):
    def __init__(self, *args, **kwargs):
        """Reactor spectra offequilibrium correction
        Applicable to ILL based 235U/239Pu/241Pu spectra (Huber, Schreckenbach)

        Chages since v04:
        - add ROOT input support
        """
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')
        self.offeq_raw_spectra = dict()
        self._load_data()

    @staticmethod
    def _provides(cfg):
            return ('offeq_scale',), ('offeq_correction',)

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
                _offeq_energy, _offeq_spectra = map(C.Points, self.offeq_raw_spectra[iso])
                _offeq_energy.points.setLabel("Original energies for offeq spectrum of {}".format(iso))
            except KeyError:
            # U238 doesn't have offequilibrium correction so just pass 1.
                if iso != 'U238':
                    raise
                passthrough = C.Identity(labels='Nominal {0} spectrum in {1} reactor'.format(iso, reac))
                self.context.objects[name] = passthrough
                dummy = C.Identity() #just to serve 1 input
                self.set_input('offeq_correction', idx, dummy.single_input(), argument_number=0)
                self.set_input('offeq_correction', idx, passthrough.single_input(), argument_number=1)
                self.set_output("offeq_correction", idx, passthrough.single())
                continue

            offeq_spectra = C.InterpLinear(labels='Correction for {} spectra'.format(iso))
            offeq_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            offeq_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

            insegment = offeq_spectra.transformations.front()
            insegment.setLabel("Offequilibrium segments")

            interpolator_trans = offeq_spectra.transformations.back()
            interpolator_trans.setLabel("Interpolated spectral correction for {}".format(iso))

            passthrough = C.Identity(labels="Nominal {0} spectrum in {1} reactor".format(iso, reac))
            _offeq_energy >> (insegment.edges, interpolator_trans.x)
            _offeq_spectra >> interpolator_trans.y

            # Enu
            self.set_input('offeq_correction', idx, (insegment.points, interpolator_trans.newx),
                            argument_number=0)
            # Anue spectra
            self.set_input('offeq_correction', idx, ( passthrough.single_input()), argument_number=1)

            par_name = "offeq_scale"
            self.reqparameter(par_name, idx, central=1., relsigma=0.3,
                    labels="Offequilibrium norm for reactor {1} and iso "
                    "{0}".format(iso, reac))
            self.reqparameter("dummy_scale", idx, central=1,
                    fixed=True, labels="Dummy weight for reactor {1} and iso "
                    "{0} for offeq correction".format(iso, reac))

            snap = C.Snapshot(passthrough.single(), labels='Snapshot of {} spectra in reac {}'.format(iso, reac))

            prod = C.Product(labels='Product of initial {} spectra and '
                             'offequilibrium corr in {} reactor'.format(iso, reac))
            prod.multiply(interpolator_trans.single())
            prod.multiply(snap.single())


            outputs = [passthrough.single(), prod.single()]
            weights = ['.'.join(("dummy_scale", idx.current_format())),
                       '.'.join((par_name, idx.current_format()))]

            with self.namespace:
                final_sum = C.WeightedSum(weights, outputs, labels='Corrected to offequilibrium '
                            '{0} spectrum in {1} reactor'.format(iso, reac))


            self.context.objects[name] = final_sum
            self.set_output("offeq_correction", idx, final_sum.single())


    def define_variables(self):
        pass
