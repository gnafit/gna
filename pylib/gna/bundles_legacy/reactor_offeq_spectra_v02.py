# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import gna.constructors as C
import numpy as np
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d
from gna.env import env

class reactor_offeq_spectra_v02(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        super(reactor_offeq_spectra_v02, self).__init__( *args, **kwargs )

        self.offeq_raw_spectra = dict()
        self.enu_trans = None

        self.make_anue_energy_bins()
        self.init_indices()
        self.load_data()

    def make_anue_energy_bins(self):
        bin_centers = (self.cfg.edges[1:] + self.cfg.edges[:-1])/2
        edges = C.Points(bin_centers)
        try:
            self.ns = env.get("ibd")
        except KeyError:
            from gna.parameters.ibd import reqparameters
            self.ns = env.ns("_offeq_ibd")
            reqparameters(self.ns)

        with self.ns:
            self.ibd = R.IbdZeroOrder()
        self.ibd.Enu.inputs(edges)



    def load_data(self):
        """Read raw input spectra"""
        data_template = self.cfg.offeq_data
        for isotope in self.idx.get_subset('i'):
            iso_name, = isotope.current_values()
            datapath = data_template.format(isotope=iso_name)
            try:
                self.offeq_raw_spectra[iso_name] = np.loadtxt(datapath, unpack=True)
            except IOError:
                pass
            assert len(self.offeq_raw_spectra) != 0, "No data loaded"

    def build(self):
        for isotope in self.idx.get_subset('i'):
            iso_name, = isotope.current_values()
            try:
                energy, spectra = self.offeq_raw_spectra[iso_name]
            except KeyError:
                continue

            f = interp1d( energy, spectra, bounds_error=False, fill_value=0.0 )
            bin_number = len(self.cfg.edges) - 1
            offeq_histo = C.Histogram(self.cfg.edges, f(self.ibd.Enu.data()))
            name = "offeq_cor."+isotope.current_format()

            offeq_histo.hist.setLabel("Offeq spectra for isotope {}".format(iso_name))

            self.objects[name] = offeq_histo
            self.set_output(offeq_histo.hist.hist, name, isotope)


    def define_variables(self):
        for it in self.idx:
            name = "offeq_scale" + it.current_format()
            iso, reac = it.current_values()
            self.common_namespace.reqparameter(name, central=1, relsigma=0.5,
                    label="Offequilibrium norm for reactor {1} and iso "
                    "{0}".format(iso, reac))
