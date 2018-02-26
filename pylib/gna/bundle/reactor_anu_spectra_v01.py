# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *

class reactor_anu_spectra_v01(TransformationBundle):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = True
    def __init__(self, **kwargs):
        kwargs['namespaces'] = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        super(reactor_anu_spectra_v01, self).__init__( **kwargs )

    def build(self):
        self.load_data()

        model_edges_t = C.Points( self.model_edges )
        self.objects['edges'] = model_edges_t
        for ns in self.namespaces:
            print(ns.name)

            # ie = R.InterpExpo(self.cfg.strategy['underflow'], self.cfg.strategy['overflow'])
            # ie.interpolate(segments_t, fcn_t, points_t)

    def load_data(self):
        self.spectra = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
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
                    self.spectra[ns.name] = data
                    break
            else:
                raise Exception('Failed to load the spectrum for %s'%ns.name)

        self.model_edges = N.ascontiguousarray( self.cfg.edges )
        if self.debug:
            print( 'Bin edges:', self.edges )




