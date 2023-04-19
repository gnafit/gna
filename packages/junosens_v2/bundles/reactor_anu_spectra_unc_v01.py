from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.bundle import TransformationBundle
from scipy.interpolate import interp1d
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from tools.root_helpers import TFileContext
from tools.data_load import read_object_auto

class reactor_anu_spectra_unc_v01(TransformationBundle):
    '''Antineutrino spectral uncertainties v01

    Based on reactor_anu_spectra_v06

    Configuration options:
    - name                                       - output names
    - ename                                      - output name for the interpolation energy
    - filename                                   - input file names (list), root/txt
    - edges                                      - array with edges for interpolation
    - varmode             = 'log'|'plain'        - parameters may be simple multipliers with central=1 or the power of e with central=0
    - varname                                    - parmeter names
    - ns_name                                    - namespace name to keep parameters in
    '''
    short_names = dict(U5  = 'U235', U8  = 'U238', Pu9 = 'Pu239', Pu1 = 'Pu241')
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.debug = self.cfg.get('debug', False)

        self.load_data()

    @staticmethod
    def _provides(cfg):
        return (), tuple(cfg.name+s for s in ('_scale', '_corrected'))+(cfg.ename,)

    def build(self):
        self.model_edges_t = C.Points(self.model_edges, labels='Spectral uncertainty edges')
        edges = self.model_edges_t.single()
        self.set_output(self.cfg.ename, None, edges)

        self.unity = C.FillLike(1.0, labels='Unity')
        edges >> self.unity
        unity = self.unity.single()

        self.objects=[]

        name1 = self.cfg.name+'_scale'
        name2 = self.cfg.name+'_corrected'
        for i, it in enumerate(self.nidx_major):
            isotope, = it.current_values()

            #
            # Create the uncertainty correction
            #
            with self.reac_ns(isotope):
                delta_uncor = C.VarArray(self.variables_uncor[isotope], labels=f'δ uncor. {isotope}')
            delta_cor   = self.cfg.varname_cor

            sigma_uncor = C.Points(self.uncertainties['uncor'][isotope], labels=f'σ uncor. {isotope}')
            sigma_cor   = C.Points(self.uncertainties['cor'][isotope], labels=f'σ cor. {isotope}')

            ds_uncor = C.Product((delta_uncor.single(), sigma_uncor.single()), labels=f'δσ uncor. {isotope}')
            with self.reac_ns:
                ds_cor = C.WeightedSum([delta_cor], [sigma_cor.single()], labels=f'δσ cor. {isotope}')

            scale_uncor = C.Sum([unity, ds_uncor], labels=f'Scale uncor. {isotope}')
            scale_cor = C.Sum([unity, ds_cor], labels=f'Scale cor. {isotope}')

            scale = C.Product([scale_uncor, scale_cor], labels=f'Scale uncert. {isotope}')
            self.set_output(name1, it, scale.single())

            #
            # Create the interpolator
            #
            interp = R.InterpExpo(labels=(f'InSeg {isotope} unc. corrected', f'Interp {isotope} unc. corrected'))
            sampler = interp.transformations.front()
            interpolator = interp.transformations.back()
            edges >> (sampler.inputs.edges, interpolator.x)

            self.set_input(name2, it, (sampler.inputs.points, interpolator.inputs.newx), argument_number=0)
            self.set_input(name2, it, interpolator.inputs.y, argument_number=1)
            self.set_output(name2, it, interpolator.interp)

            self.objects.extend((delta_uncor, delta_cor, sigma_uncor, sigma_cor,
                                 ds_uncor, ds_cor, scale_uncor, scale_cor, scale,
                                 interp))

    def load_data(self):
        """Read raw input spectra"""
        self.uncertainties_in = dict( cor={}, uncor={} )
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')
        for it in self.nidx_major:
            isotope, = it.current_values()
            for tp, target in self.uncertainties_in.items():
                data = self.load_file(self.cfg.filename, dtype, type=tp, isotope=isotope)
                target[isotope] = data
                if self.debug:
                    print(f'Load {tp} for {isotope}:', data)

        """Read parametrization edges"""
        self.model_edges = np.ascontiguousarray(self.cfg.edges, dtype='d')
        model_widths = self.model_edges[1:]-self.model_edges[:-1]
        model_widths = np.concatenate([model_widths, [model_widths[-1]]])
        if self.debug:
            print( 'Bin edges:', self.model_edges )
            print( 'Bin widths:', model_widths )

        #
        # Compute values at the parametrization points
        # Rescale errors according to the bin width
        # (at this point assume the spectrum constant within bin width)
        #
        # Absolute: sigma_i     = sigma*sqrt(w_i/w)
        # Absolute: var_i       = var*w_i/w
        # Relative: sigma_rel_i = sigma_rel*sqrt(w/w_i)
        # Relative: var_rel_i   = var_rel*w/w_i
        #
        self.uncertainties = dict(cor={}, uncor={})
        for tp, source in self.uncertainties_in.items():
            for isotope, (eorig, unc) in source.items():
                widths = eorig[1:]-eorig[:-1]
                widths = np.concatenate([widths, [widths[-1]]])
                var = unc**2
                var_scaled = var*widths
                var_scaled_interp = interp1d(eorig, var_scaled, bounds_error=False, fill_value='extrapolate' )

                scaled_new = var_scaled_interp(self.model_edges)
                var_new = scaled_new/model_widths
                unc_new=var_new**0.5
                unc_new[unc_new>0.4]=0.4
                self.uncertainties[tp][isotope]=unc_new
                if self.debug:
                    print(f'Original {tp} for {isotope}:', unc)
                    print(f'Rescaled {tp} for {isotope}:', unc_new)

    def define_variables(self):
        self.reac_ns = self.namespace(self.cfg.ns_name) if self.cfg.get('ns_name') else self.namespace

        fixed = self.cfg.get('fixed', False)
        self.variable_cor=self.reqparameter(self.cfg.varname_cor, None, central=0.0, sigma=1.0,
                                            label='anue flux correlated unc.', fixed=fixed,
                                            namespace=self.reac_ns)

        self.variables_uncor={}
        for it in self.nidx_major:
            isotope, = it.current_values()
            isons = self.reac_ns(isotope)
            variables = self.variables_uncor.setdefault(isotope, [])
            for index in range(self.model_edges.size):
                energy=self.model_edges[index]
                name = self.cfg.varname_uncor.format(index=index)
                var=self.reqparameter(name, None, central=0.0, sigma=1.0, fixed=fixed,
                        label=f'anue flux uncorrelated unc. {isotope} {energy:.3f} MeV',
                        namespace=isons)
                variables.append(name)

    def load_file(self, filenames, dtype, **kwargs):
        if isinstance(filenames, str) and filenames.endswith('.root'):
            return self.load_root(filenames, **kwargs)

        for fmt in filenames:
            fname = fmt.format(**kwargs)
            try:
                data = np.loadtxt(fname, dtype, unpack=True)
            except:
                pass

            else:
                if self.debug:
                    print( kwargs, fname )
                    print( data )
                return data

        raise KeyError('Failed to load file for '+str(kwargs))

    # def load_root(self, filename, isotope):
        # try:
            # objectnamefmt = self.cfg['objectnamefmt']
        # except KeyError:
            # raise Exception('Need to provide `objectnamefmt` format for reading a ROOT file')

        # objectname = objectnamefmt.format(isotope=isotope)
        # return read_object_auto(filename, name=objectname, verbose=True, suffix=' [{}]'.format(isotope))
