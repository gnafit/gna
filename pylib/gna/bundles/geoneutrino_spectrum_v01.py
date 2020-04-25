# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from gna.env import env

class geoneutrino_spectrum_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.isotopes=['238U', '232Th']
        self.data={}
        if not self.cfg.data:
            raise Exception('Data path not provided')
        if not '{' in self.cfg.data:
            raise Exception('Data path does not contain format: {isotope}')
        self._load_data()

    @staticmethod
    def _provides(cfg):
        return ('geoneutrino_spectrum',), ()

    def _load_data(self):
        """Read raw input spectra"""
        for iso in self.isotopes:
            path = self.cfg.data.format(isotope=iso)
            try:
                datum = np.loadtxt(path, unpack=True)
            except:
                raise Exception('Unable to read input file: '+path)

            self.data[iso]=datum

    def build(self):
        self.inputs = {}
        for k, v in self.data.items():
            x = C.Points(v[0])
            y = C.Points(v[0])

            interp = C.InterpLinear(labels='{} geo-neutrino spectra'.format(k))
            interp.set_underflow_strategy(R.GNA.Interpolation.Stratety.Constant)
            interp.set_overflow_strategy(R.GNA.Interpolation.Stratety.Constant)

        # for idx in self.nidx.iterate():
            # offeq_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            # offeq_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

            # insegment = offeq_spectra.transformations.front()
            # insegment.setLabel("Segments")

            # interpolator_trans = offeq_spectra.transformations.back()
            # interpolator_trans.setLabel("Interpolated spectral correction for {}".format(iso))

            # ones = C.FillLike(1., labels="Nominal spectra for {}".format(iso))
            # _offeq_energy >> (insegment.edges, interpolator_trans.x)
            # _offeq_spectra >> interpolator_trans.y

            # self.set_input('offeq_correction', idx, (insegment.points,
                            # interpolator_trans.newx, ones.single_input()), argument_number=0)

            # par_name = "offeq_scale"
            # self.reqparameter(par_name, idx, central=1., relsigma=0.3,
                    # labels="Offequilibrium norm for reactor {1} and iso "
                    # "{0}".format(iso, reac))
            # self.reqparameter("dummy_scale", idx, central=1,
                    # fixed=True, labels="Dummy weight for reactor {1} and iso "
                    # "{0} for offeq correction".format(iso, reac))


            # outputs = [ones.single(), interpolator_trans.single()]
            # weights = ['.'.join(("dummy_scale", idx.current_format())),
                       # '.'.join((par_name, idx.current_format()))]

            # with self.namespace:
                # final_sum = C.WeightedSum(weights, outputs, labels='Offeq correction to '
                            # '{0} spectrum in {1} reactor'.format(iso, reac))


            # self.context.objects[name] = final_sum
            # self.set_output("offeq_correction", idx, final_sum.single())


    def define_variables(self):
        pass
