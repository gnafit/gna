# -*- coding: utf-8 -*-

from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R

@patchROOTClass(R.OutputDescriptor, '__str__')
def OutputDescriptor____str__(self):
    return '[out] {}: {:s}'.format(self.name(), self.check() and self.datatype() or 'invalid')

