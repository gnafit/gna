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
        kwargs['namespaces'] = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        super(reactor_anu_spectra_v01, self).__init__( **kwargs )

    def build(self):
        self.load_data()

        model_edges_t = C.Points( self.model_edges, ns=self.common_namespace )
        self.objects['edges'] = model_edges_t

        with self.common_namespace:
            corrections_raw_t = R.VarArray(C.stdvector(self.variables), ns=self.common_namespace)
        if self.cfg.varmode=='log':
            corrections_t = R.Exp(ns=self.common_namespace)
            corrections_t.exp.points( corrections_raw_t )
            self.objects['corrections_log'] = corrections_raw_t
        else:
            corrections_t = corrections_raw_t
        self.objects['corrections'] = corrections_t

        newx = self.shared.points
        for ns in self.namespaces:
            spectrum_raw_t = C.Points( self.spectra[ns.name], ns=self.common_namespace )

            spectrum_t = R.Product(ns=self.common_namespace)
            spectrum_t.multiply( spectrum_raw_t )
            spectrum_t.multiply( corrections_t )

            interp_expo_t = R.InterpExpo(self.cfg.strategy['underflow'], self.cfg.strategy['overflow'], ns=self.common_namespace)
            interp_expo_t.interpolate(model_edges_t, spectrum_t, newx)

            """Store data"""
            self.objects[('spectrum_raw', ns.name)] = spectrum_raw_t
            self.objects[('spectrum', ns.name)]     = spectrum_t
            self.objects[('interp', ns.name)]       = interp_expo_t
            self.transformations_out[ns.name]       = interp_expo_t.interp
            self.outputs[ns.name]                   = interp_expo_t.interp.interp

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for ns in self.namespaces:
            for format in self.cfg.filename:
                fname = format.format(iso=ns.name)
                try:
                    data = N.loadtxt(fname, dtype, unpack=True)
                except:
                    pass
                else:
                    if self.debug:
                        print( ns.name, fname )
                        print( data )
                    self.spectra_raw[ns.name] = data
                    break
            else:
                raise Exception('Failed to load the spectrum for %s'%ns.name)

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

    def define_variables(self):
        varmode = self.cfg.varmode
        if not varmode in ['log', 'plain']:
            raise Exception('Unknown varmode (should be log or plain): '+str(varmode))

        self.variables=[]
        for i in range(self.cfg.edges.size):
            name = self.cfg.varname.format( index=i )
            self.variables.append(name)

            if varmode=='log':
                self.common_namespace.reqparameter( name, central=0.0, sigma=0.1 )
            else:
                self.common_namespace.reqparameter( name, central=1.0, sigma=0.1 )
