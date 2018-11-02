# -*- coding: utf-8 -*-

import ROOT as R
from gna.bindings import patchROOTClass
import numpy as N

@patchROOTClass(R.DataType.Hist('DataType'), '__str__')
def DataType__Hist____str__(self):
    dt=self.cast()
    if len(dt.shape)==1:
        edges = N.asanyarray(dt.edges)
        if edges.size<2:
            return 'hist, {:3d} bins, edges undefined'.format(dt.shape[0])

        width = edges[1:]-edges[:-1]
        if N.allclose(width, width[0]):
            suffix='width {}'.format(width[0])
        else:
            suffix='variable width'

        return 'hist, {:3d} bins, edges {}->{}, {}'.format(dt.shape[0], edges[0], edges[-1], suffix)
    elif len(dt.shape)==2:
        edges1 = N.asanyarray(dt.edgesNd[0])
        edges2 = N.asanyarray(dt.edgesNd[1])
        if edges1.size<2 or edges2.size<2:
            return 'hist, {:3d} bins, edges undefined'.format(dt.shape[0])

        # width1 = edges1[1:]-edges1[:-1]
        # width2 = edges2[1:]-edges2[:-1]
        # if N.allclose(width, width[0]):
            # suffix='width {}'.format(width[0])
        # else:
            # suffix='variable width'

        return 'hist2d, {:3d}x{:3d}={:d} bins, edges {}->{} and {}->{}'.format(
                dt.shape[0], dt.shape[1], dt.size(), edges1[0], edges1[-1], edges2[0], edges2[-1])

    return 'histogram, undefined'

@patchROOTClass(R.DataType.Points('DataType'), '__str__')
def DataType__Points____str__(self):
    dt=self.cast()
    if len(dt.shape):
        return 'array {}d, shape {}, size {:3d}'.format(len(dt.shape), 'x'.join((str(i) for i in dt.shape)), dt.size())

    return 'array, undefined'

@patchROOTClass
def DataType____str__(self):
    if self.defined():
        if self.kind==1:
            return str(self.points())
        elif self.kind==2:
            return str(self.hist())

        return 'datatype, unsupported'

    return 'datatype, undefined'
