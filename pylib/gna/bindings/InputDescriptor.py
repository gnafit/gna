# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from gna.bindings import patchROOTClass, DataType, provided_precisions, OutputDescriptor
import ROOT as R
from printing import printl, nextlevel

classes = [R.InputDescriptorT(ft, ft) for ft in provided_precisions]

@patchROOTClass(classes, '__str__')
def InputDescriptor____str__(self, **kwargs):
    if self.bound():
        return '[in]  {} <- {!s}'.format(self.name(), OutputDescriptor.OutputDescriptor____str__(self.output(), **kwargs))
    else:
        return '[in]  {} <- ...'.format(self.name())

@patchROOTClass(classes, 'print')
def InputDescriptor__print(self, **kwargs):
    printl(InputDescriptor____str__(self, **kwargs))

for ft in provided_precisions:
    InputHandle = R.TransformationTypes.InputHandleT(ft)
    del InputHandle.__lshift__

@patchROOTClass(classes, '__gt__')
@patchROOTClass(classes, '__lt__')
def InputDescriptor______cmp__(a, b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
