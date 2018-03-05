# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_snf_spectra_v01(TransformationBundle):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = False
    def __init__(self, **kwargs):
        self.isotopes = kwargs['namespaces'] = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        super(reactor_snf_spectra_v01, self).__init__( **kwargs )

        # self.load_data()

    def build(self):
        pass

    def load_data(self):
        # """Read raw input spectra"""
        # self.spectra_raw = OrderedDict()
        # dtype = [ ('enu', 'd'), ('yield', 'd') ]
        # if self.debug:
            # print('Load files:')
        # for ns in self.namespaces:
            # data = self.load_file(self.cfg.filename, dtype, isotope=ns.name)
            # self.spectra_raw[ns.name] = data

        # """Read parametrization edges"""
        # self.model_edges = N.ascontiguousarray( self.cfg.edges, dtype='d' )
        # if self.debug:
            # print( 'Bin edges:', self.model_edges )

        # """Compute the values of spectra on the parametrization"""
        # self.spectra=OrderedDict()
        # for name, (x, y) in self.spectra_raw.items():
            # f = interp1d( x, N.log(y), bounds_error=True )
            # model = N.exp(f(self.model_edges))
            # self.spectra[name] = model
        pass

    def define_variables(self):
        for reactor in self.cfg.reactors:
            ns = self.common_namespace(reactor)
            for isotope in self.isotopes:
                pname = 'ffrac_{}'.format(isotope)
                par = ns.reqparameter( pname, central=self.cfg.fission_fractions[isotope], relsigma=0.1 )
                par.setLabel( '{isotope} fission fraction at {reactor}'.format(isotope=isotope, reactor=reactor) )

            for isotope in self.isotopes:
                pname = 'ffrac_{}'.format(isotope)
                par = ns.reqparameter( pname+'_fixed', central=self.cfg.fission_fractions[isotope], relsigma=0.1, fixed=True )
                par.setLabel( '{isotope} fission fraction at {reactor} (const)'.format(isotope=isotope, reactor=reactor) )

