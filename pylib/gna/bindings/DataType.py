# -*- coding: utf-8 -*-

import ROOT as R
from gna.bindings import patchROOTClass

@patchROOTClass(R.DataType.Hist('DataType'), '__str__')
def DataType__Hist____str__(self):
    dt=self.cast()
    if len(dt.shape):
        return 'hist, {:3d} bins, {}->{}'.format(dt.shape[0], dt.edges[0], dt.edges[-1])

    return 'histogram, undefined'

@patchROOTClass(R.DataType.Points('DataType'), '__str__')
def DataType__Points____str__(self):
    dt=self.cast()
    if len(dt.shape):
        return 'array {}d, {}={:3d}'.format(len(dt.shape), 'x'.join(dt.shape), dt.size())

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
