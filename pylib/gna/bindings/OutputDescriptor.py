# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass, DataType, provided_precisions
import ROOT as R
from printing import printl, nextlevel
import types

classes = [R.OutputDescriptorT(ft,ft) for ft in provided_precisions]

@patchROOTClass(classes, '__str__')
def OutputDescriptor____str__(self):
    return '[out] {:s}: {:s}'.format(self.name(), self.check() and self.datatype() or 'invalid')

@patchROOTClass(classes, 'print')
def OutputDescriptor__print(self):
    printl(str(self))

@patchROOTClass(classes, 'single')
def OutputDescriptor__single(self):
    return self

@patchROOTClass(classes, '__rshift__')
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

@patchROOTClass(classes, '__rlshift__')
def OutputDescriptor______rlshift__(output, inputs):
    OutputDescriptor______rshift__(output, inputs)

@patchROOTClass(classes, '__gt__')
@patchROOTClass(classes, '__lt__')
def OutputDescriptor______cmp__(a,b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
