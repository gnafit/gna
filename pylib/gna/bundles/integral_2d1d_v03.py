# Reimplementation of integral_2d1d_v01 for up to date bundles
# Reimplementation of integral_2d1d_v02 with Integrator21GL instead of legacy GaussLegendre
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class integral_2d1d_v03(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__( self, *args, **kwargs )
        self.check_cfg()

    @staticmethod
    def _provides(cfg):
        var0, var1 = cfg.variables
        return (), ('integral', var0, var1, var0+'_centers', var0+'_edges', var0+'_hist', var0+'_mesh', var1+'_mesh')

    def check_cfg(self):
        if not 'name' in self.cfg:
            pkey = self.cfg.parent_key()
            if not pkey:
                raise Exception('"name" option is not provided for integral_1d_v01')
            self.cfg.name = pkey

        try:
            self.edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        except:
            raise Exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        try:
            self.xorders = N.ascontiguousarray(self.cfg.xorders, dtype='P')
        except:
            raise Exception('Invalid xorders definition: {!r}'.format(self.cfg.xorders))

        if len(self.cfg.variables)!=2:
            raise Exception('Two vairables should be provided')

    def build(self):
        if self.xorders.size>1:
            if self.xorders.size+1 != self.edges.size:
                raise Exception('Incompartible edges and xorders definition:\n    {!r}\n    {!r}'.format(self.edges, self.xorders))
            self.integrator = R.Integrator21GL(self.edges.size-1, self.xorders, self.edges, self.cfg.yorder, -1.0, 1.0)
        else:
            self.integrator = R.Integrator21GL(self.edges.size-1, int(self.xorders[0]), self.edges, self.cfg.yorder, -1.0, 1.0)
        self.integrator.points.setLabel('GL sampler (2d)')

        self.integrator.points.x.setLabel(self.cfg.variables[0])
        self.integrator.points.xedges.setLabel('%s edges'%self.cfg.variables[0])
        self.integrator.points.xcenters.setLabel('{} bin centers'.format(self.cfg.variables[0]))
        self.integrator.points.y.setLabel(self.cfg.variables[1])

        self.set_output(self.cfg.variables[0],            None, self.integrator.points.x)
        self.set_output('{}_edges'.format(self.cfg.variables[0]), None, self.integrator.points.xedges)
        self.set_output('{}_centers'.format(self.cfg.variables[0]),  None, self.integrator.points.xcenters)
        self.set_output('{}_hist'.format(self.cfg.variables[0]),  None, self.integrator.points.xhist)
        self.set_output('{}_mesh'.format(self.cfg.variables[0]),  None, self.integrator.points.xmesh)
        self.set_output('{}_mesh'.format(self.cfg.variables[1]),  None, self.integrator.points.ymesh)
        self.set_output(self.cfg.variables[1],            None, self.integrator.points.y)

        hist = self.integrator.hist
        for i, it in enumerate(self.nidx):
            if i:
                hist = self.integrator.add_transformation()
            hist.setLabel(it.current_format('{name} {autoindex} (GL)', name='Integral'))

            self.set_input('integral', it, self.integrator.add_input(), argument_number=0)
            self.set_output('integral', it, hist.outputs.back())

    def define_variables(self):
        pass
