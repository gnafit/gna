# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.InputDescriptor, '__str__')
def InputDescriptor____str__(self):
    if self.materialized():
        return '[in]  {} -> {!s}'.format(self.name(), self.output())
    else:
        return '[in]  {} -> ...'.format(self.name())

@patchROOTClass
def InputDescriptor__print(self):
    printl(str(self))
