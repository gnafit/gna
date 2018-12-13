# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.TransformationDescriptor, '__str__')
def TransformationDescriptor____str__(self):
    return '[trans] {}: {:d} input(s), {:d} output(s)'.format(self.name(), self.inputs.size(), self.outputs.size())

@patchROOTClass
def TransformationDescriptor__print(self):
    printl(str(self))
    with nextlevel():
        for i, inp in enumerate(self.inputs.itervalues()):
            printl('{:2d}'.format(i), end=' ')
            inp.print()
        for i, o in enumerate(self.outputs.itervalues()):
            printl('{:2d}'.format(i), end=' ')
            o.print()

@patchROOTClass
def TransformationDescriptor__single(self):
    outputs = self.outputs
    if outputs.size()!=1:
        raise Exception('Can not call single() on transformation %s with %i outputs', self.name(), outputs.size())

    return outputs.front()

@patchROOTClass
def TransformationDescriptor__single_input(self):
    inputs = self.inputs
    if inputs.size()!=1:
        raise Exception('Can not call single_input() on transformation %s with %i inputs', self.name(), inputs.size())

    return inputs.front()
