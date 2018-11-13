# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.GNAObject, '__str__')
def GNAObject____str__(self):
    return '[obj] {}: {:d} transformation(s)'.format(self.__class__.__name__, self.transformations.size())

@patchROOTClass
def GNAObject__print(self):
    printl(str(self))
    if self.transformations.size():
        with nextlevel():
            # printl('Transformations:')
            for i, t in enumerate(self.transformations.itervalues()):
                t.print()

