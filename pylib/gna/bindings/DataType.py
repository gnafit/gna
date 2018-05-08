# -*- coding: utf-8 -*-

import ROOT as R
from gna.bindings import patchROOTClass
import numpy as N

@patchROOTClass(R.DataType.Hist('DataType'), '__str__')
def DataType__Hist____str__(self):
    dt=self.cast()
    if len(dt.shape):
        edges = N.asanyarray(dt.edges)
        width = edges[1:]-edges[:-1]
        if (N.fabs(width-width[0])/width<1.e-9).all():
            suffix='width {}'.format(width[0])
        else:
            suffix='variable width'

        return 'hist, {:3d} bins, edges {}->{}, {}'.format(dt.shape[0], edges[0], edges[-1], suffix)

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
