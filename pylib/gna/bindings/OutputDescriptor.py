# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.OutputDescriptor, '__str__')
def OutputDescriptor____str__(self):
    return '[out] {:s}: {:s}'.format(self.name(), self.check() and self.datatype() or 'invalid')

@patchROOTClass
def OutputDescriptor__print(self):
    printl(str(self))

