# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_fission_fractions_const_v01(TransformationBundle):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    def __init__(self, **kwargs):
        self.isotopes = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        self.reactors = kwargs['namespaces'] = kwargs['cfg'].reactors
        super(reactor_fission_fractions_const_v01, self).__init__( **kwargs )

    def define_variables(self):
        for reactor in self.cfg.reactors:
            ns = self.common_namespace(reactor)
            for isotope in self.isotopes:
                pname = 'ffrac_{}'.format(isotope)
                par = ns.reqparameter( pname, central=self.cfg.fission_fractions[isotope], relsigma=0.1 )
                par.setLabel( '{isotope} fission fraction at {reactor}'.format(isotope=isotope, reactor=reactor) )

        for isotope in self.isotopes:
            pname = 'ffrac_{}'.format(isotope)
            par = self.common_namespace.reqparameter( pname+'_fixed', central=self.cfg.fission_fractions[isotope], relsigma=0.1, fixed=True )
            par.setLabel( '{isotope} fission fraction at {reactor} (const)'.format(isotope=isotope, reactor=reactor) )

