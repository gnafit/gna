from load import ROOT as R
import gna.constructors as C
import numpy as N
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_spectra_v02(TransformationBundleLegacy):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = False
    def __init__(self, cfg, *args, **kwargs):
        if not 'indices' in cfg:
            raise Exception('Invalid configuration: missing "indices" field')

        self.idx = cfg.indices
        from gna.expression import NIndex
        if not isinstance(self.idx, NIndex):
            raise Exception('reac_anu_spectra_v02 should be called from within expression')

        if self.idx.ndim()!=1:
            raise Exception('reac_anu_spectra_v02 supports only 1d indexing, got %i'%self.indx.ndim())

        self.isotopes = kwargs['namespaces'] = [it.current_values()[0] for it in self.idx]

        super(reactor_anu_spectra_v02, self).__init__( cfg, *args, **kwargs )

        self.load_data()

    def build(self):
        model_edges_t = C.Points( self.model_edges, ns=self.common_namespace )
        model_edges_t.points.setLabel('E0 (bin edges)')
        self.objects['edges'] = model_edges_t
        self.shared.reactor_anu_edges = model_edges_t.single()

        self.corrections=None
        if self.cfg.get('corrections', None):
            self.corrections, = execute_bundles(cfg=self.cfg.corrections, shared=self.shared)

        self.interp_expo = interp_expo = R.InterpExpo(ns=self.common_namespace)
        sampler = interp_expo.transformations.front()
        model_edges_t >> sampler.inputs.edges
        sampler_input = sampler.inputs.points
        interp_expo.bind_transformations(False)

        interp_expo_t = interp_expo.transformations.back()
        for i, it in enumerate(self.idx):
            isotope = it.current_values()[0]

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

            if i>0:
                interp_expo_t = interp_expo.add_transformation(False)
                interp_expo.bind_transformations(False)

            model_edges_t >> interp_expo_t.inputs.x
            interp_output = interp_expo.add_input(spectrum_t)
            interp_input  = interp_expo_t.inputs.newx

            if i>0:
                self.set_input(interp_input, self.cfg.name, it, clone=0)
            else:
                self.set_input((sampler_input, interp_input), self.cfg.name, it, clone=0)

            interp_expo_t.setLabel('S(E):\n'+isotope)

            """Store data"""
            self.set_output(interp_output, self.cfg.name, it)

            self.objects[('spectrum', isotope)]     = spectrum_t

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = dict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for isotope in self.isotopes:
            data = self.load_file(self.cfg.filename, dtype, isotope=isotope)
            self.spectra_raw[isotope] = data

        """Read parametrization edges"""
        self.model_edges = N.ascontiguousarray( self.cfg.edges, dtype='d' )
        if self.debug:
            print( 'Bin edges:', self.model_edges )

        """Compute the values of spectra on the parametrization"""
        self.spectra=dict()
        self.shared.reactor_anu_fcn=dict()
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
