
from load import ROOT as R
import gna.constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_spectra_v01(TransformationBundleLegacy):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = False
    def __init__(self, *args, **kwargs):
        self.isotopes = kwargs['namespaces'] = [self.short_names.get(s,s) for s in kwargs['cfg'].isotopes]
        super(reactor_anu_spectra_v01, self).__init__( *args, **kwargs )

        self.load_data()

    def build(self):
        model_edges_t = C.Points( self.model_edges, ns=self.common_namespace )
        model_edges_t.points.setLabel('E0 (model)')
        self.objects['edges'] = model_edges_t
        self.shared.reactor_anu_edges = model_edges_t.single()

        self.corrections=None
        if self.cfg.get('corrections', None):
            self.corrections, = execute_bundles(cfg=self.cfg.corrections, shared=self.shared)

        newx = self.shared.points
        segments_t=None
        for isotope in self.isotopes:
            ns = self.common_namespace(isotope)
            spectrum_raw_t = C.Points( self.spectra[isotope], ns=self.common_namespace )
            spectrum_raw_t.points.setLabel('S0(E0):\n'+isotope)
            self.objects[('spectrum_raw', isotope)] = spectrum_raw_t

            if self.corrections:
                spectrum_t = R.Product(ns=self.common_namespace)
                spectrum_t.multiply( spectrum_raw_t )
                for corr in self.corrections.bundles.values():
                    spectrum_t.multiply( corr.outputs[isotope] )
                spectrum_t.product.setLabel('S(E0):\n'+isotope)
            else:
                spectrum_t = spectrum_raw_t

            interp_expo_t = R.InterpExpoSorted(self.cfg.strategy['underflow'], self.cfg.strategy['overflow'], ns=self.common_namespace)
            interp_expo_t.interp.setLabel('S(E):\n'+isotope)
            if segments_t:
                interp_expo_t.interpolate(segments_t, model_edges_t, spectrum_t, newx)
            else:
                interp_expo_t.interpolate(model_edges_t, spectrum_t, newx)
                segments_t = interp_expo_t.segments

            """Store data"""
            self.objects[('spectrum', isotope)]     = spectrum_t
            self.objects[('interp', isotope)]       = interp_expo_t
            self.transformations_out[isotope]       = interp_expo_t.interp
            self.outputs[isotope]                   = interp_expo_t.interp.interp

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for ns in self.namespaces:
            data = self.load_file(self.cfg.filename, dtype, isotope=ns.name)
            self.spectra_raw[ns.name] = data

        """Read parametrization edges"""
        self.model_edges = N.ascontiguousarray( self.cfg.edges, dtype='d' )
        if self.debug:
            print( 'Bin edges:', self.model_edges )

        """Compute the values of spectra on the parametrization"""
        self.spectra=OrderedDict()
        self.shared.reactor_anu_fcn=OrderedDict()
        fcns = self.shared.reactor_anu_fcn
        for name, (x, y) in self.spectra_raw.items():
            f = interp1d( x, N.log(y), bounds_error=True )
            fcns[name] = lambda e: N.exp(f(e))
            model = N.exp(f(self.model_edges))
            self.spectra[name] = model

    def define_variables(self):
        pass

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
