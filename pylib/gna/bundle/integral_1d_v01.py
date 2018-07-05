# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class integral_1d_v01(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

        if not 'name' in self.cfg:
            pkey = self.cfg.parent_key()
            if not pkey:
                raise Exception('"name" option is not provided for integral_1d_v01')
            self.cfg.name = pkey

        self.idx = self.cfg.indices
        from gna.expression import NIndex
        if not isinstance(self.idx, NIndex):
            self.idx = NIndex(fromlist=self.cfg.indices)

    def build(self):
        try:
            self.edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        except:
            raise Exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        try:
            self.orders = N.ascontiguousarray(self.cfg.orders, dtype='P')
        except:
            raise Exception('Invalid orders definition: {!r}'.format(self.cfg.orders))

        if self.orders.size>1:
            if self.orders.size+1 != self.edges.size:
                raise Exception('Incompartible edges and orders definition:\n    {!r}\n    {!r}'.format(self.edges, self.orders))
            self.integrator = R.GaussLegendre(self.edges, self.orders, self.edges.size-1)
        else:
            self.integrator = R.GaussLegendre(self.edges, int(self.orders[0]), self.edges.size-1)
        self.integrator.points.setLabel('integrator 1d')

        if self.context:
            self.context.set_output(self.integrator.points.x,      self.cfg.variable)
            self.context.set_output(self.integrator.points.xedges, '%s_edges'%self.cfg.variable)

        for i, it in enumerate(self.idx.iterate()):
            hist = R.GaussLegendreHist(self.integrator)
            hist.hist.setLabel( it.current_format('hist{autoindex}') )

            if self.context:
                self.context.set_input( hist.hist.f,    self.cfg.name, it, clone=0)
                self.context.set_output(hist.hist.hist, self.cfg.name, it)

    def define_variables(self):
        pass

