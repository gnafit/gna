# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass, DataType, provided_precisions
import ROOT as R
from printing import printl, nextlevel
import types
import numpy as np

classes = tuple(R.OutputDescriptorT(ft,ft) for ft in provided_precisions)
classes_input = tuple(R.InputDescriptorT(ft,ft) for ft in provided_precisions)
classes_object = tuple(R.GNAObjectT(ft,ft) for ft in provided_precisions)
classes_td = tuple(R.TransformationDescriptorT(ft,ft) for ft in provided_precisions)

@patchROOTClass(classes, '__str__')
def OutputDescriptor____str__(self, **kwargs):
    ret = '[out] {}: {}'.format(self.name(), self.check() and self.datatype() or 'invalid')
    data, sl = kwargs.pop('data', False), kwargs.pop('slice', slice(None))
    if data and self.check() and self.datatype():
        values = str(self.data()[sl])
        ret = ret+': '+values+'\n'
    return ret

@patchROOTClass(classes, 'print')
def OutputDescriptor__print(self, **kwargs):
    printl(OutputDescriptor____str__(self, **kwargs))

@patchROOTClass(classes, 'single')
def OutputDescriptor__single(self):
    return self

@patchROOTClass(classes, '__rshift__')
def OutputDescriptor______rshift__(output, inputs):
    if isinstance(inputs, classes_input):
        inputs.connect(output)
    elif isinstance(inputs, (list, tuple,types.GeneratorType)):
        for inp in inputs:
            OutputDescriptor______rshift__(output, inp)
    elif isinstance(inputs, classes_object+classes_td):
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

@patchROOTClass(classes, 'data')
@patchROOTClass(classes, '__call__')
def OutputDescriptor__data(self):
    buf = self.__data_orig()
    datatype = self.datatype()
    return np.frombuffer(buf, count=datatype.size(), dtype=buf.typecode).reshape(datatype.shape, order='F')
