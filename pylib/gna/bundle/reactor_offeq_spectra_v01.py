# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_offeq_spectra_v01(TransformationBundle):
    parname = 'offeq_scale'
    debug = False
    def __init__(self, **kwargs):
        super(reactor_offeq_spectra_v01, self).__init__( **kwargs )

        self.load_data()

    def build(self):
        parnames = []
        offeqs = OrderedDict()
        for name, corr in self.offeq.items():
            corr_o = C.Points(corr, ns=self.common_namespace)
            corr_o.points.setLabel( '{}\n(offeq corr)'.format(name) )
            self.objects[('offeq_corr', name)] = corr_o

            specfcn = self.shared.reactor_anu_fcn[name]
            spec = specfcn(self.edges)
            spec_o = C.Points(spec, ns=self.common_namespace)
            spec_o.points.setLabel( '{}'.format(name) )
            self.objects[('isotopes', name)] = spec_o

            offeq_o = R.Product()
            offeq_o.product.setLabel( '{}\n(offeq)'.format(name) )
            offeq_o.multiply(corr_o, spec_o)

            self.objects[('offeq', name)] = offeq_o
            offeqs[name]=offeq_o.single()

        sumnames = C.stdvector(['offeq'])
        sumweights = C.stdvector([self.parname])
        for ns in self.namespaces:
            for isotope, corr in offeqs.items():
                with ns:
                    spec_o = R.WeightedSum(sumnames, sumweights, ns=ns)

                self.objects[('spec_scaled', ns.name, isotope)]=spec_o

                spec_o.sum.setLabel('Offeq spectrum\n{} at {}'.format(isotope, ns.name))
                spec_o.sum.inputs['offeq'](corr)

                self.transformations_out[(ns.name,isotope)] = spec_o.sum
                self.outputs[(ns.name,isotope)]             = spec_o.sum.sum

    def load_data(self):
        """Read raw input spectra"""
        self.edges = self.cfg.get('edges', None)
        if self.edges is None:
            self.edges = self.shared.reactor_anu_edges.data()

        skip = self.cfg.get('skip', [])
        self.offeq=OrderedDict()
        dtype = [ ('enu', 'd'), ('correction', 'd') ]
        for name, specfcn in self.shared.reactor_anu_fcn.items():
            if name in skip:
                continue
            fname = self.cfg.filename.format(isotope=name)
            if self.debug:
                print('Load:', fname)
            x, y = N.loadtxt(fname, dtype, unpack=True)

            f = interp1d( x, y, bounds_error=False, fill_value=0.0 )

            self.offeq[name] = f(self.edges)

    def define_variables(self):
        for ns in self.namespaces:
            reactor = ns.name
            ns = self.common_namespace(reactor)

            par = ns.reqparameter( self.parname, cfg=self.cfg.norm )
            par.setLabel( 'SNF contribution for {reactor}'.format(reactor=reactor) )
