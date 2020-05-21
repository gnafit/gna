# -*- coding: utf-8 -*-

from __future__ import absolute_import
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

        return 'hist2d, {:d}x{:d}={:d} bins, edges {}->{} and {}->{}'.format(
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

@patchROOTClass
def DataType__isHist(self):
    return self.defined() and self.kind==2

@patchROOTClass
def DataType__isPoints(self):
    return self.defined() and self.kind==1

@patchROOTClass
def DataType____eq__(self, other):
    if self.kind!=other.kind:
        return False

    if list(self.shape)!=list(other.shape):
        return False

    if self.kind!=2:
        return True

    for (e1, e2) in zip(self.edgesNd, other.edgesNd):
        edges1 = N.asanyarray(e1)
        edges2 = N.asanyarray(e2)

        if not N.allclose(edges1, edges2, rtol=0, atol=1.e-14):
            return False

    return True

@patchROOTClass
def DataType____ne__(self, other):
    return not DataType____eq__(self, other)
