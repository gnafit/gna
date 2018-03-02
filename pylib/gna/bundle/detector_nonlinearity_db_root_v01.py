# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as N
import constructors as C
from converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from gna.env import env, namespace
from gna.configurator import NestedDict
from gna.bundle import *

class detector_nonlinearity_db_root_v01(TransformationBundle):
    debug = False
    def __init__(self, **kwargs):
        super(detector_nonlinearity_db_root_v01, self).__init__( **kwargs )

        self.edges = kwargs.pop('edges', self.shared.get('edges', None))
        if not self.edges:
            raise Exception('detector_nonlinearity_db_root_v01 expects bin edges to be passed as argument or shared')
        self.storage=NestedDict()
        self.pars=NestedDict()

        self.transformations_in = self.transformations_out

    def build_graphs( self, graphs ):
        #
        # Interpolate curves on the default binning
        # (extrapolate as well)
        #
        newx = self.edges.data()
        self.storage['edges'] = newx
        newy = []
        for xy, name in zip(graphs, self.cfg.names):
            f = interpolate( xy, newx )
            newy.append(f)
            self.storage[name] = f.copy()

        #
        # All curves but first are the corrections to the nominal
        #
        for f in newy[1:]:
            f-=newy[0]

        #
        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        #
        with self.common_namespace:
            corr_lsnl = R.WeightedSum( C.stdvector(self.cfg.names), C.stdvector(['weight_'+n for n in self.cfg.names]), ns=self.common_namespace )
        self.objects['lsnl_factor']=corr_lsnl

        for y, name in zip( newy, self.cfg.names ):
            pts = C.Points( y, ns=self.common_namespace )
            corr_lsnl.sum[name]( pts )

            self.objects[('curves', name)] = pts

        with self.common_namespace:
            for i, ns in enumerate(self.namespaces):
                with ns:
                    """Uncorrelated between detectors part of the energy nonlinearity factor
                    correlated part multiplicated by the scale factor"""
                    labels = C.stdvector([self.pars[ns.name]])
                    corr = R.WeightedSum(labels, labels, ns=ns)
                    corr.sum.inputs[0]( corr_lsnl.sum )

                    """Finally, original bin edges multiplied by the correction factor"""
                    newe = R.Product(ns=ns)
                    newe.multiply( self.edges )
                    newe.multiply( corr.sum )

                    """Construct the nonlinearity calss"""
                    nonlin = R.HistNonlinearity(self.debug, ns=ns)
                    nonlin.set(self.edges, newe.product)

                    """Provide output transformations"""
                    self.transformations_out[ns.name] = nonlin.smear
                    self.inputs[ns.name]              = nonlin.smear.Ntrue
                    self.outputs[ns.name]             = nonlin.smear.Nvis

                    """Save intermediate transformations"""
                    self.objects[('factor', ns.name)]       = corr
                    self.objects[('edges_mod', ns.name)]    = newe
                    self.objects[('nonlinearity', ns.name)] = nonlin

                    """Define observables"""
                    self.addcfgobservable(ns, nonlin.smear.Nvis, 'nonlinearity', ignorecheck=True)

    def build(self):
        tfile = R.TFile( self.cfg.filename, 'READ' )
        if tfile.IsZombie():
            raise IOError( 'Can not read ROOT file: '+self.cfg.filename )

        graphs = [ tfile.Get( name ) for name in self.cfg.names ]
        if not all( graphs ):
            raise IOError( 'Some objects were not read from file: '+filename )

        graphs = [ get_buffers_graph(g) for g in graphs ]

        return self.build_graphs( graphs )

    def define_variables(self):
        par = self.common_namespace.reqparameter( 'weight_'+self.cfg.names[0], central=1.0, sigma=0.0, fixed=True )
        par.setLabel( 'Nominal nonlinearity curve weight' )
        for name in self.cfg.names[1:]:
            par = self.common_namespace.reqparameter( 'weight_'+name, central=0.0, sigma=1.0 )
            par.setLabel( 'Correction nonlinearity weight for '+name )

        if self.cfg.par.central!=1:
            raise Exception('Relative energy scale parameter should have central value of 1 by definition')
        for ns in self.namespaces:
            parname = self.cfg.parname.format(ns.name)
            par = self.common_namespace.reqparameter( parname, cfg=self.cfg.par )
            par.setLabel( 'Uncorrelated energy scale for '+ns.name )
            self.pars[ns.name]=parname

def interpolate( (x, y), edges):
    fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value='extrapolate' )
    res = fcn( edges )
    return res
