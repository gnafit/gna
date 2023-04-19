from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.bundle import TransformationBundle
from scipy.interpolate import interp1d
from tools.data_load import read_object_auto

from tools.schema import Schema, And, isreadable, Or, Optional
from gna.configurator import StripNestedDict

class reactor_anu_spectra_v08(TransformationBundle):
    '''Antineutrino specta model v08

    Changes since v06:
    - Add an option to set uncertainty
    - Add an option to set `spectral_parameters` constrained
    - rename `varname` to `varnamefmt`
    - rename `filename` to `filenamefmt`
    - add a configuration validator

    Configuration options:
    - name                                                          - output names
    - objectnamefmt                                                 - object name mask for the ROOT file
    - filenamefmt                                                   - input file names (list), root/txt
    - edges                                                         - array with edges for interpolation
    - spectral_parameters = 'disabled'|'free'|'fixed'|'constrained' - disable/enable parameters to modify the spectrum, parameters may be free, fixed or constrained
    - uncertainty                                                   - [optional] uncertainty on the parameter value
    - uncertainty_segment_width                                     - [optional] segment width for which the uncertainty is defined
    - varmode             = 'log'|'plain'                           - parameters may be simple multipliers with central=1 or the power of e with central=0
    - varnamefmt                                                    - parmeter names
    - ns_name                                                       - [optional] namespace name to keep parameters in
    '''
    short_names = dict(U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241')
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major') # major index - isotope

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

        self.load_data()

    _validator = Schema({
            'bundle': object,
            'name': str,
            'objectnamefmt': str,
            'filenamefmt': [str],
            'edges': Or((Or(float, int)), [Or(float, int)], np.ndarray),
            'spectral_parameters': Or('disabled', 'free', 'fixed', 'constrained'),
            'varmode': Or('log', 'plain'),
            'varnamefmt': str,
            Optional('ns_name', default=None): str,
            Optional('uncertainty', default=None): float,
            Optional('uncertainty_segment_width', default=None): float,
        })

    @staticmethod
    def _provides(cfg):
        return (), tuple(cfg.name+s for s in ('', '_enu', '_scale'))

    def build(self):
        self.objects = {}

        output_name = self.vcfg['name']
        varmode = self.vcfg['varmode']

        model_edges_t = C.Points( self.model_edges, ns=self.namespace )
        model_edges_t.points.setLabel('Spectra interpolation edges')
        self.objects['edges'] = model_edges_t
        self.set_output(output_name+'_enu', None, model_edges_t.single())

        spectral_parameters_mode = self.vcfg['spectral_parameters']
        spectral_parameters_enabled = spectral_parameters_mode!='disabled'
        if spectral_parameters_enabled:
            with self.reac_ns:
                tmp = C.VarArray(self.variables, ns=self.reac_ns, labels='{{Spec pars:|log(n_i)}}')
            if varmode == 'log':
                self.objects['npar_log'] = tmp
                free_weights = R.Exp(ns=self.reac_ns)
                self.objects['free_weights'] = free_weights
                free_weights.exp.points( tmp )
                free_weights.exp.setLabel('exp(n_i)')
            else:
                tmp.vararray.setLabel('n_i')
                free_weights = tmp
                self.objects['free_weights'] = free_weights

            self.set_output(output_name+'_scale', None, free_weights.single())

        spectra_in = self.objects['spectra_in'] = {}
        spectra_mod = self.objects['spectra_mod'] = {}
        for i, itmajor in enumerate(self.nidx_major):
            isotope, = itmajor.current_values()

            spectrum_raw_t = C.Points( self.spectra[isotope], ns=self.reac_ns )
            spectrum_raw_t.points.setLabel('%s spectrum, original'%isotope)
            spectra_in[isotope] = spectrum_raw_t

            if spectral_parameters_enabled:
                spectrum_t = C.Product(ns=self.reac_ns)
                spectrum_t.multiply( spectrum_raw_t )
                spectrum_t.multiply( free_weights.single() )
                spectrum_t.product.setLabel('%s spectrum, corrected'%isotope)
                spectra_mod[isotope]=spectrum_t

        interps = self.objects['interp'] = []
        for itminor in self.nidx_minor:
            interp_expo = R.InterpExpo(ns=self.reac_ns)
            interps.append(interp_expo)

            self.context.objects[('interp', isotope)] = spectrum_t
            sampler = interp_expo.transformations.front()
            model_edges_t >> sampler.inputs.edges
            sampler_input = sampler.inputs.points
            interp_expo_t = interp_expo.transformations.back()

            for i, itmajor in enumerate(self.nidx_major):
                isotope, = itmajor.current_values()

                it = itmajor+itminor

                spectrum_raw_t = spectra_in[isotope]

                if spectral_parameters_enabled:
                    spectrum_t = spectra_mod[isotope]
                else:
                    spectrum_t = spectra_in[isotope]

                if i>0:
                    interp_expo_t = interp_expo.add_transformation()
                interp_expo_t.setLabel(itminor.current_format('{{Interpolated %s spectrum | {autoindex}}}'%isotope))

                model_edges_t >> interp_expo_t.inputs.x
                interp_output = interp_expo.add_input(spectrum_t)
                interp_input  = interp_expo_t.inputs.newx

                if i>0:
                    self.set_input(output_name, it, interp_input, argument_number=0)
                else:
                    self.set_input(output_name, it, (sampler_input, interp_input), argument_number=0)

                self.set_output(output_name, it, interp_output)

    def load_data(self):
        """Read raw input spectra"""
        self.spectra_raw = dict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for it in self.nidx_major:
            isotope, = it.current_values()
            data = self.load_file(self.vcfg['filenamefmt'], dtype, isotope=isotope)
            self.spectra_raw[isotope] = data

        """Read parametrization edges"""
        self.model_edges = np.ascontiguousarray(self.vcfg['edges'], dtype='d')
        if self.debug:
            print( 'Bin edges:', self.model_edges )

        """Compute the values of spectra on the parametrization"""
        self.spectra = dict()
        for name, (x, y) in self.spectra_raw.items():
            f = interp1d( x, np.log(y), bounds_error=False, fill_value='extrapolate' )
            model = np.exp(f(self.model_edges))
            self.spectra[name] = model

    def define_variables(self):
        spectral_parameters_mode = self.vcfg['spectral_parameters']
        spectral_parameters_enabled = spectral_parameters_mode!='disabled'
        if not spectral_parameters_enabled:
            return

        ns_name = self.vcfg['ns_name']
        self.reac_ns = self.namespace(ns_name) if ns_name else self.namespace

        uncertainty = self.vcfg['uncertainty']
        uncertainty_segment_width = self.vcfg['uncertainty_segment_width']

        if (uncertainty is None)!=(uncertainty_segment_width is None):
            raise self.exception('The `uncertainty` and `uncertainty_segment_width` should be both defined or both undefined')

        varnamefmt = self.vcfg['varnamefmt']

        if spectral_parameters_mode=='free':
            free  = True
            fixed = False
        elif spectral_parameters_mode=='constrained':
            free  = False
            fixed = False

            if uncertainty is None:
                raise self.exception('The `uncertainty` and `uncertainty_segment_width` should be both defined for constrained parameters')
        else:
            free = False
            fixed = True
        varmode = self.vcfg['varmode']

        self.variables=[]
        for i in range(self.model_edges.size):
            name = varnamefmt.format(index=i)
            self.variables.append(name)

            energy = self.model_edges[i]
            kwargs = {'free': free, 'fixed': fixed}

            if uncertainty is not None:
                try:
                    energy_width = self.model_edges[i+1] - self.model_edges[i]
                except IndexError:
                    energy_width = self.model_edges[i] - self.model_edges[i-1]

                kwargs['sigma'] = uncertainty * (uncertainty_segment_width/energy_width)**0.5

            if varmode=='log':
                label = 'Average reactor spectrum correction for {} MeV [log]'.format(energy)
                self.reac_ns.reqparameter(name, central=0.0, label=label, **kwargs)
            else:
                label='Average reactor spectrum correction for {} MeV'.format(energy)
                self.reac_ns.reqparameter(name, central=1.0, label=label, **kwargs)

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
        objectnamefmt = self.vcfg['objectnamefmt']
        objectname = objectnamefmt.format(isotope=isotope)
        return read_object_auto(filename, name=objectname, verbose=True, suffix=' [{}]'.format(isotope))
