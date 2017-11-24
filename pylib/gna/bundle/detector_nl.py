# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as N
from load import ROOT as R
import constructors as C
from converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from scipy.interpolate import interp1d
from matplotlib import pyplot as P
from gna.env import env, namespace
from gna.bundle import *

@declare_bundle('nonlinearity_db_root_v01')
class detector_nonlinearity_db_root_v01(TransformationBundle):
    debug = False
    parname = 'escale'
    def __init__(self, edges, **kwargs):
        kwargs.setdefault( 'storage_name', 'nonlinearity')
        super(detector_nonlinearity_db_root_v01, self).__init__( **kwargs )

        self.edges = edges
        self.storage['edges'] = edges

    def build_graphs( self, graphs ):
        #
        # Interpolate curves on the default binning
        # (extrapolate as well)
        #
        newx = self.edges.data()
        self.storage('inputs')['edges'] = newx
        newy = []
        for xy, name in zip(graphs, self.cfg.names):
            f = interpolate( xy, newx )
            newy.append(f)
            self.storage('inputs')[name] = f.copy()

        #
        # All curves but first are the corrections to the nominal
        #
        for f in newy[1:]:
            f-=newy[0]

        #
        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        #
        corr_lsnl = self.storage['lsnl_factor'] = R.WeightedSum( convert(self.cfg.names, 'stdvector') )
        for y, name in zip( newy, self.cfg.names ):
            pts = C.Points( y )
            self.storage('curves')[name] = pts
            corr_lsnl.sum[name]( pts )

        output = ()
        labels = convert([self.parname], 'stdvector')
        for i, ns in enumerate(self.namespaces):
            with ns:
                #
                # Uncorrelated between detectors part of the energy nonlinearity factor
                # correlated part multiplicated by the scale factor
                #
                lstorage = self.storage('escale_%s'%ns.name if ns.name else 'escale')
                corr = lstorage['factor'] = R.WeightedSum( labels, labels )
                corr.sum['escale']( corr_lsnl.sum )

                #
                # Finally, original bin edges multiplied by the correction factor
                #
                newe = lstorage['edges_mod'] = R.Product()
                newe.multiply( self.edges )
                newe.multiply( corr.sum )

                #
                # Construct the nonlinearity calss
                #
                nonlin = lstorage['nonlinearity'] = R.HistNonlinearity( self.debug )
                nonlin.set( self.edges, newe.product )
                output+=nonlin,

        return output

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
        self.common_namespace.reqparameter( 'weight_'+self.cfg.names[0], central=1.0, sigma=0.0, fixed=True )
        for name in self.cfg.names[1:]:
            self.common_namespace.reqparameter( 'weight_'+name, central=0.0, sigma=1.0 )

        for ns in self.namespaces:
            ns.reqparameter( self.parname, central=1.0, uncertainty=self.cfg.uncertainty, uncertainty_type=self.cfg.uncertainty_type )

def interpolate( (x, y), edges):
    fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value='extrapolate' )
    res = fcn( edges )
    return res
