# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from gna.env import env

class geoneutrino_spectrum_v01(TransformationBundle):
    isotopes=['U238', 'Th232']
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.data={}
        if not self.cfg.data:
            raise Exception('Data path not provided')
        if not '{' in self.cfg.data:
            raise Exception('Data path does not contain format: {isotope}')
        self._load_data()

    @staticmethod
    def _provides(cfg):
        return ('geonu_spectrum_{}'.format(k) for k in self.isotopes), ('geonu_norm_{}'.format(k) for k in self.isotopes)

    def _load_data(self):
        """Read raw input spectra"""
        for iso in self.isotopes:
            iso_reverse = iso[-3:]+iso[:-3]
            path = self.cfg.data.format(isotope=iso_reverse)
            try:
                datum = np.loadtxt(path, unpack=True)
            except:
                raise Exception('Unable to read input file: '+path)

            datum[0]*=1.e-3 # keV to MeV
            datum[1]*=1.e3  # 1/keV to 1/MeV
            self.data[iso]=datum

    def build(self):
        self.inputs = {}
        self.outputs = {}
        self.points = {}
        self.interp = {}
        for k, v in self.data.items():
            x = C.Points(v[0], labels='{} geo-nu X0'.format(k))
            y = C.Points(v[1], labels='{} geo-nu Y0'.format(k))

            interp = C.InterpLinear(labels=('{} X insegment'.format(k), '{} geo-neutrino spectra'.format(k)))
            interp.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            interp.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
            interp.set_fill_value(0.0)
            interp.setXY(x,y)

            self.points[k] = (x, y)
            self.interp[k] = interp

        for idx in self.nidx.iterate():
            for k in self.data.keys():
                interp = self.interp[k]
                self.set_input('geonu_spectrum_{}'.format(k), idx, (interp.insegment.points, interp.interp.newx), argument_number=0)
                self.set_output('geonu_spectrum_{}'.format(k), idx, interp.interp.interp)

    def define_variables(self):
        for iso in self.isotopes:
            self.reqparameter('geonu_norm_%s'%iso, None, central=1.0, free=True, label='%s normalization factor'%iso)

