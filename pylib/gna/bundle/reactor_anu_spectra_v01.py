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

        model_edges_t = C.Points( self.model_edges )
        self.objects['edges'] = model_edges_t

        newx = self.shared.points
        for ns in self.namespaces:
            spectrum_t = C.Points( self.spectra[ns.name] )

            ie = R.InterpExpo(self.cfg.strategy['underflow'], self.cfg.strategy['overflow'])
            ie.interpolate(model_edges_t, spectrum_t.single(), newx)

            """Store data"""
            self.objects[('interp', ns.name)] = ie
            self.transformations_out[ns.name] = ie.interp
            self.outputs[ns.name] = ie.interp.interp

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






