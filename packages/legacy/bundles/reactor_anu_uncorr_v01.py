from load import ROOT as R
import gna.constructors as C
import numpy as N
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_uncorr_v01(TransformationBundleLegacy):
    debug = False
    def __init__(self, *args, **kwargs):
        super(reactor_anu_uncorr_v01, self).__init__( *args, **kwargs )

        self.edges = self.shared.reactor_anu_edges.data()
        self.bundles=dict( self=self )

        self.load_data()

    def build(self):
        uncpars = dict()
        for name, vars in self.uncorr_vars.items():
            with self.common_namespace:
                uncpar_t = C.VarArray(vars, ns=self.common_namespace)
            uncpar_t.vararray.setLabel('Uncorr correction:\n'+name)
            uncpars[name]=uncpar_t

            self.objects[('uncorrelated_correction', name)] = uncpar_t
            self.transformations_out[name] = uncpar_t.transformations[0]
            self.outputs[name]             = uncpar_t.single()

    def define_variables(self):
        self.uncorr_vars=dict()
        for ns in self.namespaces:
            isotope = ns.name
            uncfcn = interp1d( *self.uncertainties_uncorr[isotope] )
            for i in range(self.edges.size):
                name = self.cfg.uncnames.format( isotope=isotope, index=i )
                self.uncorr_vars.setdefault(isotope, []).append(name)

                en=self.edges[i]
                var = self.common_namespace.reqparameter( name, central=1.0, sigma=uncfcn(en) )
                var.setLabel('Uncorrelated {} anu spectrum correction for {} MeV'.format(isotope, en))

    def load_data(self):
        self.uncertainties_uncorr = dict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')

        for ns in self.namespaces:
            unc_uncorr = self.load_file(self.cfg.uncertainties, dtype, isotope=ns.name, mode='uncorr')
            self.uncertainties_uncorr[ns.name] = unc_uncorr

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
