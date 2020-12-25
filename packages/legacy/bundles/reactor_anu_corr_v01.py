
from load import ROOT as R
import gna.constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_corr_v01(TransformationBundleLegacy):
    debug = False
    def __init__(self, *args, **kwargs):
        super(reactor_anu_corr_v01, self).__init__( *args, **kwargs )

        self.edges = self.shared.reactor_anu_edges.data()
        self.bundles=OrderedDict( self=self )

        self.load_data()

    def build(self):
        corrpars = OrderedDict()
        for name, vars in self.corr_vars.items():
            with self.common_namespace:
                corr_sigma_t = C.VarArray(vars, ns=self.common_namespace)
                corrpar_t = R.WeightedSum(1.0, C.stdvector([self.cfg.uncname]), C.stdvector(['offset']))
                corrpar_i = corrpar_t.sum.inputs
                corrpar_i['offset']( corr_sigma_t )

            corr_sigma_t.vararray.setLabel('Corr unc:\n'+name)
            corrpar_t.sum.setLabel('Corr correction:\n'+name)
            corrpars[name]=corrpar_t

            self.objects[('correlated_sigma', name)] = corr_sigma_t
            self.objects[('correlated_correction', name)] = corrpar_t

            self.transformations_out[name] = corrpar_t.transformations[0]
            self.outputs[name]             = corrpar_t.single()

    def define_variables(self):
        self.common_namespace.reqparameter( self.cfg.uncname, central=0.0, sigma=1.0, label='Correlated reactor anu spectrum correction (offset)'  )

        self.corr_vars=OrderedDict()
        for ns in self.namespaces:
            isotope=ns.name
            corrfcn = interp1d( *self.uncertainties_corr[isotope] )
            for i in range(self.edges.size):
                name = self.cfg.parnames.format( isotope=isotope, index=i )
                self.corr_vars.setdefault(isotope, []).append(name)

                en = self.edges[i]
                var = self.common_namespace.reqparameter( name, central=corrfcn(en), sigma=0.1, fixed=True )
                var.setLabel('Correlated {} anu spectrum correction sigma for {} MeV'.format(isotope, en))

    def load_data(self):
        self.uncertainties_corr = OrderedDict()
        dtype = [ ('enu', 'd'), ('yield', 'd') ]
        if self.debug:
            print('Load files:')

        for ns in self.namespaces:
            unc_corr = self.load_file(self.cfg.uncertainties, dtype, isotope=ns.name, mode='corr')
            self.uncertainties_corr[ns.name] = unc_corr

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
