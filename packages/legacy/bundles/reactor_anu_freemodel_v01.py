
from load import ROOT as R
import gna.constructors as C
import numpy as N
from collections import OrderedDict
from gna.bundle import *
from scipy.interpolate import interp1d

class reactor_anu_freemodel_v01(TransformationBundleLegacy):
    debug = False
    def __init__(self, *args, **kwargs):
        super(reactor_anu_freemodel_v01, self).__init__( *args, **kwargs )

        self.edges = self.shared.reactor_anu_edges.data()

        self.bundles=OrderedDict( self=self )

    def build(self):
        with self.common_namespace:
            npar_raw_t = C.VarArray(self.variables, ns=self.common_namespace)

        nsname = self.common_namespace.name
        if self.cfg.varmode=='log':
            npar_raw_t.vararray.setLabel('Spec pars:\nlog(n_i)')
            npar_t = R.Exp(ns=self.common_namespace)
            npar_t.exp.points( npar_raw_t )
            npar_t.exp.setLabel('n_i')
            self.objects['npar_log'] = npar_raw_t

        else:
            npar_raw_t.vararray.setLabel('n_i')
            npar_t = npar_raw_t

        for ns in self.namespaces:
            """Store data"""
            self.transformations_out[ns.name] = npar_t.transformations[0]
            self.outputs[ns.name]             = npar_t.single()

        self.objects['corrections'] = npar_t

    def define_variables(self):
        varmode = self.cfg.varmode
        if not varmode in ['log', 'plain']:
            raise Exception('Unknown varmode (should be log or plain): '+str(varmode))

        self.variables=[]
        for i in range(self.edges.size):
            name = self.cfg.varname.format( index=i )
            self.variables.append(name)

            if varmode=='log':
                var=self.common_namespace.reqparameter( name, central=0.0, sigma=N.inf )
                var.setLabel('Average reactor spectrum correction for {} MeV [log]'.format(self.edges[i]))
            else:
                var=self.common_namespace.reqparameter( name, central=1.0, sigma=N.inf )
                var.setLabel('Average reactor spectrum correction for {} MeV'.format(self.edges[i]))
