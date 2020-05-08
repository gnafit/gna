# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from gna.env import env

class reactor_snf_spectra_v04(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self.snf_raw_data = dict()
        self._load_data()

    @staticmethod
    def _provides(cfg):
            return ('snf_scale',), ('snf_correction',)


    def _load_data(self):
        """Read raw input spectra. Can be either average spectrum or individual
        for each reactor. """
        def _load_average(datapath):
            self.snf_raw_data['average'] = np.loadtxt(datapath, unpack=True)

        def _load_by_reactor(template):
            for reac_idx in self.nidx.get_subset('r'):
                reac, = reac_idx.current_values()
                key = reac_idx.names()[0]
                datapath = template.format(**{key: reac})
                self.snf_raw_data[reac] = np.loadtxt(datapath, unpack=True)

        try:
            datapath = self.cfg.snf_average_spectra
            _load_average(datapath)
        except KeyError:
            try:
                data_template = self.cfg.snf_pools_spectra
                _load_by_reactor(data_template)
            except KeyError:
                raise

    def build(self):
        for idx in self.nidx:
            reac, = idx.current_values()
            name = "snf_correction" + idx.current_format()
            try:
                _snf_energy, _snf_spectra = map(C.Points, self.snf_raw_data[reac])
            except KeyError:
            # U238 doesn't have offequilibrium correction so just pass 1.
                _snf_energy, _snf_spectra = map(C.Points, self.snf_raw_data['average'])

            _snf_energy.points.setLabel("Original energies for SNF spectrum of {}".format(reac))

            snf_spectra = C.InterpLinear(labels='Correction for spectra in {}'.format(reac))
            snf_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            snf_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

            insegment = snf_spectra.transformations.front()
            insegment.setLabel("Segments")

            interpolator_trans = snf_spectra.transformations.back()
            interpolator_trans.setLabel("Interpolated SNF correction for {}".format(reac))

            passthrough = C.Identity(labels="Nominal spectra for {}".format(reac))
            _snf_energy >> (insegment.edges, interpolator_trans.x)
            _snf_spectra >> interpolator_trans.y

            self.set_input('snf_correction', idx, (insegment.points, interpolator_trans.newx), argument_number=0)
            self.set_input('snf_correction', idx, (passthrough.single_input()), argument_number=1)

            snap = C.Snapshot(passthrough.single(), labels='Snapshot of nominal spectra for SNF in {}'.format(reac))
            product = C.Product(outputs=[snap.single(), interpolator_trans.single()],
                                labels='Product of nominal spectrum to SNF correction in {}'.format(reac))

            par_name = "snf_scale"
            self.reqparameter(par_name, idx, central=1., relsigma=0.3,
                              labels="SNF norm for reactor {0}".format(reac))

            outputs = [product.single()]
            weights = ['.'.join((par_name, idx.current_format()))]

            with self.namespace:
                final_sum = C.WeightedSum(weights, outputs, labels='SNF spectrum from {0} reactor'.format(reac))

            self.context.objects[name] = final_sum
            self.set_output("snf_correction", idx, final_sum.single())

    def define_variables(self):
        pass
