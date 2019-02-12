# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass, DataType, provided_precisions
import ROOT as R
from printing import printl, nextlevel

classes = [R.InputDescriptorT(ft,ft) for ft in provided_precisions]

@patchROOTClass(classes, '__str__')
def InputDescriptor____str__(self):
    if self.bound():
        return '[in]  {} <- {!s}'.format(self.name(), self.output())
    else:
        return '[in]  {} <- ...'.format(self.name())

@patchROOTClass(classes, 'print')
def InputDescriptor__print(self):
    printl(str(self))

for ft in provided_precisions:
    InputHandle = R.TransformationTypes.InputHandleT(ft)
    del InputHandle.__lshift__

@patchROOTClass(classes, '__gt__')
@patchROOTClass(classes, '__lt__')
def InputDescriptor______cmp__(a,b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
