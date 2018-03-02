# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_spectra_v01(TransformationBundle):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = False
    def __init__(self, **kwargs):
        self.isotopes = kwargs['namespaces'] = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        super(reactor_anu_spectra_v01, self).__init__( **kwargs )

        self.load_data()

    def build(self):
        model_edges_t = C.Points( self.model_edges, ns=self.common_namespace )
        model_edges_t.points.setLabel('E0 (model)')
        self.objects['edges'] = model_edges_t
        self.shared.reactor_anu_edges = model_edges_t.single()

        self.corrections=None
        if 'corrections' in self.cfg:
            self.corrections, = execute_bundle(cfg=self.cfg.corrections, shared=self.shared)

        corrpars = OrderedDict()
        for name, vars in self.corr_vars.items():
            with self.common_namespace:
                corr_sigma_t = R.VarArray(C.stdvector(vars), ns=self.common_namespace)
                corrpar_t = R.WeightedSum(1.0, C.stdvector(['offset']), C.stdvector([self.cfg.corrname]))
                corrpar_i = corrpar_t.sum.inputs
                corrpar_i['offset']( corr_sigma_t )

            corr_sigma_t.vararray.setLabel('Corr unc:\n'+name)
            corrpar_t.sum.setLabel('Corr correction:\n'+name)
            corrpars[name]=corrpar_t

            self.objects[('correlated_sigma', name)] = corr_sigma_t
            self.objects[('correlated_correction', name)] = corrpar_t

        newx = self.shared.points
        segments_t=None
        for ns in self.namespaces:
            isotope=ns.name
            spectrum_raw_t = C.Points( self.spectra[isotope], ns=self.common_namespace )
            spectrum_raw_t.points.setLabel('S0(E0):\n'+isotope)

            spectrum_t = R.Product(ns=self.common_namespace)
            spectrum_t.multiply( spectrum_raw_t )
            for corr in self.corrections.bundles.values():
                spectrum_t.multiply( corr.outputs[isotope] )

            spectrum_t.multiply( corrpars[isotope] )
            spectrum_t.product.setLabel('S(E0):\n'+isotope)

            interp_expo_t = R.InterpExpo(self.cfg.strategy['underflow'], self.cfg.strategy['overflow'], ns=self.common_namespace)
            interp_expo_t.interp.setLabel('S(E):\n'+isotope)
            if segments_t:
                interp_expo_t.interpolate(segments_t, model_edges_t, spectrum_t, newx)
            else:
                interp_expo_t.interpolate(model_edges_t, spectrum_t, newx)
                segments_t = interp_expo_t.segments

            """Store data"""
            self.objects[('spectrum_raw', isotope)] = spectrum_raw_t
            self.objects[('spectrum', isotope)]     = spectrum_t
            self.objects[('interp', isotope)]       = interp_expo_t
            self.transformations_out[isotope]       = interp_expo_t.interp
            self.outputs[isotope]                   = interp_expo_t.interp.interp

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = OrderedDict()
        self.uncertainties_corr = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for ns in self.namespaces:
            data = self.load_file(self.cfg.filename, dtype, isotope=ns.name)
            self.spectra_raw[ns.name] = data

            unc_corr = self.load_file(self.cfg.uncertainties, dtype, isotope=ns.name, mode='corr')
            self.uncertainties_corr[ns.name] = unc_corr

        """Read parametrization edges"""
        self.model_edges = N.ascontiguousarray( self.cfg.edges, dtype='d' )
        if self.debug:
            print( 'Bin edges:', self.model_edges )

        """Compute the values of spectra on the parametrization"""
        self.spectra=OrderedDict()
        for name, (x, y) in self.spectra_raw.items():
            f = interp1d( x, N.log(y), bounds_error=True )
            model = N.exp(f(self.model_edges))
            self.spectra[name] = model

        """Read the uncertainties edges"""
        self.unc_edges=self.cfg.uncedges
        if self.unc_edges=='same':
            self.unc_edges = self.model_edges
        else:
            """If the differ from the model edges, check that they contain all the model edges"""
            self.unc_edges=N.ascontiguousarray(self.unc_edges, dtype='d')

            raise Exception('not implemented')

    def define_variables(self):
        self.common_namespace.reqparameter( self.cfg.corrname, central=0.0, sigma=1.0, label='Correlated reactor anu spectrum correction (offset)'  )

        self.corr_vars=OrderedDict()
        for isotope in self.isotopes:
            corrfcn = interp1d( *self.uncertainties_corr[isotope] )
            for i in range(self.unc_edges.size):
                name = self.cfg.corrnames.format( isotope=isotope, index=i )
                self.corr_vars.setdefault(isotope, []).append(name)

                en = self.model_edges[i]
                var = self.common_namespace.reqparameter( name, central=corrfcn(en), sigma=0.1, fixed=True )
                var.setLabel('Correlated {} anu spectrum correction sigma for {} MeV'.format(isotope, en))

    def load_file(self, filenames, dtype, **kwargs):
        for format in filenames:
            fname = format.format(**kwargs)
            try:
                data = N.loadtxt(fname, dtype, unpack=True)
            except:
                pass
            else:
                if self.debug:
                    print( kwargs, fname )
                    print( data )
                return data

        raise Exception('Failed to load file for '+str(kwargs))
