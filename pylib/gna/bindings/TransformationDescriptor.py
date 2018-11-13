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
    if self.inputs.size():
        with nextlevel():
            # printl('Outputs:')
            for i, inp in enumerate(self.inputs.itervalues()):
                inp.print()
    if self.outputs.size():
        with nextlevel():
            # printl('Outputs:')
            for i, o in enumerate(self.outputs.itervalues()):
                o.print()
