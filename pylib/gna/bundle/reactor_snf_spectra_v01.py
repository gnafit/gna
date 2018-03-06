# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_snf_spectra_v01(TransformationBundle):
    parname = 'snf_scale'
    def __init__(self, **kwargs):
        super(reactor_snf_spectra_v01, self).__init__( **kwargs )

        self.load_data()

    def build(self):
        parnames = []
        specs = OrderedDict()
        for name, specfcn in self.shared.reactor_anu_fcn.items():
            spec = specfcn(self.edges)
            spec_o = C.Points(spec, ns=self.common_namespace)
            spec_o.points.setLabel( '{}\n(SNF)'.format(name) )
            self.objects[('isotopes', name)] = spec_o

            specs[name]=spec_o.single()
            parnames.append('ffrac_{}_fixed'.format(name))

        ratio_o=C.Points(self.ratio, ns=self.common_namespace)
        ratio_o.points.setLabel('SNF\nratio')
        self.objects['snf_ratio']=ratio_o

        isonames = C.stdvector(list(specs.keys()))
        parnames = C.stdvector(parnames)
        sumnames = C.stdvector(['snf'])
        sumweights = C.stdvector([self.parname])

        with self.common_namespace:
            spec_o = R.WeightedSum(isonames, parnames, ns=self.common_namespace )

        self.objects['anue_spec_raw']=spec_o
        spec_o.sum.setLabel('anue spectrum\n(for SNF)')
        for name, spec in specs.items():
            spec_o.sum.inputs[name]( spec )

        snf_spec_o = R.Product()
        snf_spec_o.multiply(spec_o)
        snf_spec_o.multiplyaratio_o)
        snf_spec_o.product.setLabel('SNF spectrum')
        self.objects['snf_spec_raw']=snf_spec_o

        for ns in self.namespaces:
            with ns:
                spec_scaled_o = R.WeightedSum(sumnames, sumweights, ns=ns)

            self.objects[('spec_scaled', ns.name)]=spec_scaled_o

            spec_scaled_o.sum.setLabel('SNF spectrum\n{}'.format(ns.name))
            spec_scaled_o.sum.inputs['snf'](snf_spec_o)

            self.transformations_out[ns.name] = spec_scaled_o.sum
            self.outputs[ns.name]             = spec_scaled_o.sum.sum

    def load_data(self):
        """Read raw input spectra"""
        dtype = [ ('enu', 'd'), ('ratio', 'd') ]
        x, y = N.loadtxt(self.cfg.filename, dtype, unpack=True)

        self.edges = self.cfg.get('edges', None)
        if self.edges is None:
            self.edges = self.shared.reactor_anu_edges.data()

        f = interp1d( x, y, bounds_error=False, fill_value=0.0 )
        self.ratio = f(self.edges)

    def define_variables(self):
        for ns in self.namespaces:
            reactor = ns.name
            ns = self.common_namespace(reactor)

            par = ns.reqparameter( self.parname, cfg=self.cfg.norm )
            par.setLabel( 'SNF contribution for {reactor}'.format(reactor=reactor) )
