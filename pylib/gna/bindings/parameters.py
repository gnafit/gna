from __future__ import print_function
from gna.bindings import patchROOTClass, DataType, provided_precisions, ROOT

variable  = ROOT.variable('void')
parameter  = ROOT.parameter('void')

@patchROOTClass(variable, 'cast')
def variable__cast(self):
    """Cast variable<void> to the appropriate type"""
    return ROOT.variable(self.typeName())(self)

@patchROOTClass(parameter, 'cast')
def parameter__cast(self):
    """Cast parameter<void> to the appropriate type"""
    return ROOT.parameter(self.typeName())(self)
