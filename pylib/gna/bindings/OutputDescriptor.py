# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel
import types

@patchROOTClass(R.OutputDescriptor, '__str__')
def OutputDescriptor____str__(self):
    return '[out] {:s}: {:s}'.format(self.name(), self.check() and self.datatype() or 'invalid')

@patchROOTClass
def OutputDescriptor__print(self):
    printl(str(self))

@patchROOTClass
def OutputDescriptor__single(self):
    return self

@patchROOTClass(R.OutputDescriptor, '__rshift__')
def OutputDescriptor______rshift__(output, inputs):
    if isinstance(inputs, R.InputDescriptor):
        inputs.connect(output)
    elif isinstance(inputs, (list, tuple,types.GeneratorType)):
        for inp in inputs:
            OutputDescriptor______rshift__(output, inp)
    elif isinstance(inputs, (R.GNAObject, R.TransformationDescriptor)):
        OutputDescriptor______rshift__(output, inputs.single_input())
    else:
        raise Exception('Failed to connect {} to {}'.format(output.name(), inputs))

@patchROOTClass(R.OutputDescriptor, '__rlshift__')
def OutputDescriptor______rlshift__(output, inputs):
    OutputDescriptor______rshift__(output, inputs)

@patchROOTClass(R.OutputDescriptor, '__gt__')
@patchROOTClass(R.OutputDescriptor, '__lt__')
def OutputDescriptor______cmp__(a,b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
