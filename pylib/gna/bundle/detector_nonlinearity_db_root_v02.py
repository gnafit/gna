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
from collections import OrderedDict
from gna.bundle import *

class detector_nonlinearity_db_root_v02(TransformationBundle):
    debug = False
    def __init__(self, **kwargs):
        super(detector_nonlinearity_db_root_v02, self).__init__( **kwargs )

        self.init_indices()
        if self.idx.ndim()<2:
            raise Exception('detector_nonlinearity_db_root_v02 supports at least 2d indexing: detector and lsnl component')

        self.storage=NestedDict()
        self.pars=NestedDict()
        self.transformations_in = self.transformations_out

    def build_graphs( self, graphs ):
        #
        # Interpolate curves on the default binning
        # (extrapolate as well)
        #
        self.newx_out = self.context.outputs[self.cfg.edges]
        newx = self.newx_out.data()
        newy = OrderedDict()
        for xy, name in zip(graphs, self.cfg.names):
            f = interpolate( xy, newx )
            newy[name]=f
            self.storage[name] = f.copy()

        #
        # All curves but first are the corrections to the nominal
        #
        newy_values = newy.values()
        for f in newy_values[1:]:
            f-=newy_values[0]

        idxl, idxother = self.idx.split( ('l') )
        idxd, idxother = idxother.split( ('d') )

        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        for i, itl in enumerate(idxl.iterate()):
            name, = itl.current_values()
            if not name in newy:
                raise Exception('The nonlinearity curve {} is not provided'.format(name))
            y = newy[name]
            pts = C.Points( y, ns=self.common_namespace )
            if i:
                label=itl.current_format('NL correction {autoindexnd}')
            else:
                label=itl.current_format('NL nominal ({autoindexnd})')
            pts.points.setLabel(label)
            self.set_output(pts.single(), 'lsnl_component', itl)
            self.objects[('curves', name)] = pts

        with self.common_namespace:
            for i, itd in enumerate(idxd.iterate()):
                """Finally, original bin edges multiplied by the correction factor"""
                """Construct the nonlinearity calss"""
                nonlin = R.HistNonlinearity(False, self.debug)
                nonlin.matrix.setLabel(itd.current_format('NL matrix\n{autoindexnd}'))
                self.objects[('nonlinearity',)+itd.current_values()] = nonlin
                self.set_input(nonlin.matrix.Edges,         'lsnl_edges', itd, clone=0)
                self.set_input(nonlin.matrix.EdgesModified, 'lsnl_edges', itd, clone=1)

                for itother in idxother.iterate():
                    it = itd+itother
                    trans = nonlin.add()
                    trans.setLabel(it.current_format('NL\n{autoindexnd}'))

                    self.set_input(trans.Ntrue, 'lsnl', it, clone=0)
                    self.set_output(trans.Nvis, 'lsnl', it)

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
        idxl, idxother = self.idx.split( ('l') )
        idxd, idxother = idxother.split( ('d') )

        par=None
        lname = self.cfg.parnames['lsnl']
        for itl in idxl.iterate():
            name = itl.current_format('{name}{autoindex}', name = lname)
            if par is None:
                par = self.common_namespace.reqparameter(name, central=1.0, sigma=0.0, fixed=True)
                par.setLabel( itl.current_format('Nominal nonlinearity curve weight ({autoindexnd})') )
            else:
                par = self.common_namespace.reqparameter(name, central=0.0, sigma=1.0)
                par.setLabel( itl.current_format('Correction nonlinearity weight for {autoindexnd}' ))

        if self.cfg.par.central!=1:
            raise Exception('Relative energy scale parameter should have central value of 1 by definition')

        ename = self.cfg.parnames['escale']
        for it in idxd.iterate():
            parname = it.current_format('{name}{autoindex}', name=ename)
            par = self.common_namespace.reqparameter( parname, cfg=self.cfg.par )
            par.setLabel( 'Uncorrelated energy scale for '+it.current_format('{autoindexnd}') )
            self.pars[it.current_values()]=parname

def interpolate( (x, y), edges):
    fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value='extrapolate' )
    res = fcn( edges )
    return res
