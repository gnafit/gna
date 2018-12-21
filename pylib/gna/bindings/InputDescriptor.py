# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.InputDescriptor, '__str__')
def InputDescriptor____str__(self):
    if self.materialized():
        return '[in]  {} <- {!s}'.format(self.name(), self.output())
    else:
        return '[in]  {} <- ...'.format(self.name())

@patchROOTClass
def InputDescriptor__print(self):
    printl(str(self))

del R.TransformationTypes.InputHandle.__lshift__

@patchROOTClass(R.InputDescriptor, '__gt__')
@patchROOTClass(R.InputDescriptor, '__lt__')
def InputDescriptor______cmp__(a,b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
