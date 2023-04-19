from load import ROOT as R
import gna.constructors as C
import numpy as N
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict
from scipy.interpolate import interp1d
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from tools.root_helpers import TFileContext
from tools.data_load import read_object_auto

class reactor_anu_spectra_v06(TransformationBundle):
    '''Antineutrino specta model v06

    Changes since v05:
    - Make spectral parameters fixable

    Configuration options:
    - name                                       - output names
    - objectnamefmt                              - object name mask for the ROOT file
    - filename                                   - input file names (list), root/txt
    - edges                                      - array with edges for interpolation
    - spectral_parameters = False|'free'|'fixed' - disable/enable parameters to modify the spectrum, parameters may be free or fixed
    - varmode             = 'log'|'plain'        - parameters may be simple multipliers with central=1 or the power of e with central=0
    - varname                                    - parmeter names
    - ns_name                                    - namespace name to keep parameters in
    '''
    short_names = dict(U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241')
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.shared = NestedDict() # TODO: remove
        self.load_data()

    @staticmethod
    def _provides(cfg):
        return (), tuple(cfg.name+s for s in ('', '_enu', '_scale'))

    def build(self):
        model_edges_t = C.Points( self.model_edges, ns=self.namespace )
        model_edges_t.points.setLabel('Spectra interpolation edges')
        self.context.objects['edges'] = model_edges_t
        self.shared.reactor_anu_edges = model_edges_t.single()

        if self.cfg.spectral_parameters:
            with self.reac_ns:
                tmp = C.VarArray(self.variables, ns=self.reac_ns, labels='{{Spec pars:|log(n_i)}}')
            if self.cfg.varmode == 'log':
                self.context.objects['npar_log'] = tmp
                self.free_weights = R.Exp(ns=self.reac_ns)
                self.free_weights.exp.points( tmp )
                self.free_weights.exp.setLabel('exp(n_i)')
            else:
                tmp.vararray.setLabel('n_i')
                self.free_weights = tmp

            self.set_output(self.cfg.name+'_scale', None, self.free_weights.single())

        self.interp_expo = interp_expo = R.InterpExpo(ns=self.reac_ns)
        sampler = interp_expo.transformations.front()
        model_edges_t >> sampler.inputs.edges
        sampler_input = sampler.inputs.points

        interp_expo_t = interp_expo.transformations.back()
        for i, it in enumerate(self.nidx_major):
            isotope, = it.current_values()

            spectrum_raw_t = C.Points( self.spectra[isotope], ns=self.reac_ns )
            spectrum_raw_t.points.setLabel('%s spectrum, original'%isotope)
            self.context.objects[('spectrum_raw', isotope)] = spectrum_raw_t

            if self.cfg.spectral_parameters:
                spectrum_t = C.Product(ns=self.reac_ns)
                spectrum_t.multiply( spectrum_raw_t )
                spectrum_t.multiply( self.free_weights.single() )
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

        self.set_output(self.cfg.name+'_enu', None, model_edges_t.single())

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = dict()
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
        self.spectra = dict()
        self.shared.reactor_anu_fcn = dict()
        fcns = self.shared.reactor_anu_fcn
        for name, (x, y) in self.spectra_raw.items():
            f = interp1d( x, N.log(y), bounds_error=False, fill_value='extrapolate' )
            fcns[name] = lambda e: N.exp(f(e))
            model = N.exp(f(self.model_edges))
            self.spectra[name] = model

    def define_variables(self):
        self.reac_ns = self.namespace(self.cfg.ns_name) if self.cfg.get('ns_name') else self.namespace
        spectral_parameters = self.cfg.spectral_parameters
        if spectral_parameters:
            if spectral_parameters=='free':
                free = True
            else:
                free = False
            fixed = not free
            varmode = self.cfg.varmode
            if not varmode in ['log', 'plain']:
                raise KeyError('Unknown varmode {} (should be log or plain)'.format(varmode))

            self.variables=[]
            for i in range(self.model_edges.size):
                name = self.cfg.varname.format( index=i )
                self.variables.append(name)

                if varmode=='log':
                    var=self.reac_ns.reqparameter( name, central=0.0, free=free, fixed=fixed,
                            label='Average reactor spectrum correction for {} MeV [log]'.format(self.model_edges[i]))
                else:
                    var=self.reac_ns.reqparameter( name, central=1.0, free=free, fixed=fixed,
                            label='Average reactor spectrum correction for {} MeV'.format(self.model_edges[i]))
        else:
            pass

    def load_file(self, filenames, dtype, **kwargs):
        if isinstance(filenames, str) and filenames.endswith('.root'):
            return self.load_root(filenames, **kwargs)

        for fmt in filenames:
            fname = fmt.format(**kwargs)
            try:
                data = read_object_auto(fname, verbose=True, dtype=dtype)
            except:
                pass

            else:
                if self.debug:
                    print( kwargs, fname )
                    print( data )
                return data

        raise KeyError('Failed to load file for '+str(kwargs))

    def load_root(self, filename, isotope):
        try:
            objectnamefmt = self.cfg['objectnamefmt']
        except KeyError:
            raise Exception('Need to provide `objectnamefmt` format for reading a ROOT file')

        objectname = objectnamefmt.format(isotope=isotope)
        return read_object_auto(filename, name=objectname, verbose=True, suffix=' [{}]'.format(isotope))
