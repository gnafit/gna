# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
import numpy as np
from gna.bundle import *
from gna.env import env
from scipy.interpolate import interp1d
import constructors as C 

class reactor_snf_spectra_v02(TransformationBundle):
    def __init__(self, **kwargs):
        super(reactor_snf_spectra_v02, self).__init__(**kwargs)

        self.init_indices()
        self.make_anue_energy_bins()
        self.init_data()

        self.build()


    def make_anue_energy_bins(self):
        bin_centers = (self.cfg.edges[1:] + self.cfg.edges[:-1])/2
        bin_centers = C.Points(bin_centers)
        try:
            self.ns = env.get("ibd")
        except KeyError:
            from gna.parameters.ibd import reqparameters
            self.ns = env.ns("_snf_ibd")
            reqparameters(self.ns)

        with self.ns:
            self.ibd = ROOT.IbdZeroOrder()
        self.ibd.Enu.inputs(bin_centers)


    def init_data(self):
        '''Read interpolated SNF ratio from file'''  
        print(self.cfg)
        self.ratio_energy , self.ratio = np.loadtxt(self.cfg.data_path, unpack=True)
        f = interp1d(self.ratio_energy, self.ratio, bounds_error=False, fill_value=0.0 )
        self.ratio_spectrum = f(self.ibd.Enu.data())


    def build(self):
        snf_ratio = C.Histogram(self.cfg.edges, self.ratio_spectrum )

        self.objects['snf_spectra'] = snf_ratio
        self.set_output(snf_ratio.single(), "snf_ratio")
        
    def define_variables(self):
        self.common_namespace.reqparameter("snf_rate", central=0.03,
                relsigma=1., label="SNF rate normalization")
