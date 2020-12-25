
from load import ROOT as R
import gna.constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict
from scipy.interpolate import interp1d

class reactor_anu_spectra_v03(TransformationBundle):
    short_names = dict( U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241' )
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.shared = NestedDict() # TODO: remove

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        self.load_data()

        model_edges_t = C.Points( self.model_edges, ns=self.namespace )
        model_edges_t.points.setLabel('Spectra interpolation edges')
        self.context.objects['edges'] = model_edges_t
        self.shared.reactor_anu_edges = model_edges_t.single()

        self.corrections=None
        if self.cfg.get('corrections', None):
            self.corrections, = execute_bundles(cfg=self.cfg.corrections, shared=self.shared)

        self.interp_expo = interp_expo = R.InterpExpo(ns=self.namespace)
        sampler = interp_expo.transformations.front()
        model_edges_t >> sampler.inputs.edges
        sampler_input = sampler.inputs.points

        interp_expo_t = interp_expo.transformations.back()
        for i, it in enumerate(self.nidx_major):
            isotope, = it.current_values()

            spectrum_raw_t = C.Points( self.spectra[isotope], ns=self.namespace )
            spectrum_raw_t.points.setLabel('%s spectrum, original'%isotope)
            self.context.objects[('spectrum_raw', isotope)] = spectrum_raw_t

            if self.corrections:
                spectrum_t = R.Product(ns=self.namespace)
                spectrum_t.multiply( spectrum_raw_t )
                for corr in self.corrections.bundles.values():
                    spectrum_t.multiply( corr.outputs[isotope] )
                spectrum_t.product.setLabel('%s spectrum, corrected'%isotope)
            else:
                spectrum_t = spectrum_raw_t

            if i>0:
                interp_expo_t = interp_expo.add_transformation()

            model_edges_t >> interp_expo_t.inputs.x
            interp_output = interp_expo.add_input(spectrum_t)
            interp_input  = interp_expo_t.inputs.newx

            if i>0:
                self.set_input(self.cfg.name, it, interp_input, argument_number=0)
            else:
                self.set_input(self.cfg.name, it, (sampler_input, interp_input), argument_number=0)

            interp_expo_t.setLabel('%s spectrum, interpolated'%isotope)

            """Store data"""
            self.set_output(self.cfg.name, it, interp_output)

            self.context.objects[('spectrum', isotope)] = spectrum_t

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for it in self.nidx_major:
            isotope, = it.current_values()
            data = self.load_file(self.cfg.filename, dtype, isotope=isotope)
            self.spectra_raw[isotope] = data

        """Read parametrization edges"""
        self.model_edges = N.ascontiguousarray( self.cfg.edges, dtype='d' )
        if self.debug:
            print( 'Bin edges:', self.model_edges )

        """Compute the values of spectra on the parametrization"""
        self.spectra = OrderedDict()
        self.shared.reactor_anu_fcn = OrderedDict()
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
