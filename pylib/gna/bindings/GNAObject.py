# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

@patchROOTClass(R.GNAObject, '__str__')
def GNAObject____str__(self):
    return '[obj] {}: {:d} transformation(s), {:d} variables'.format(self.__class__.__name__, self.transformations.size(), self.variables.size())

@patchROOTClass
def GNAObject__print(self):
    self.printtransformations()
    if not self.variables:
        return

    if self.variables.size()>0:
        print()
        self.printvariables()

@patchROOTClass
def GNAObject__printtransformations(self):
    printl(str(self))
    if self.transformations.size():
        with nextlevel():
            # printl('Transformations:')
            for i, t in enumerate(self.transformations.itervalues()):
                printl('{:2d}'.format(i), end=' ')
                t.print()

@patchROOTClass
def GNAObject__printvariables(self):
    from gna.parameters import printer
    ns = self.currentns

    printl('Variables, relative to namespace "{}":'.format(ns.name))
    with nextlevel():
        if self.variables.size()>0:
            for name in self.variables:
                printl(ns[name].__str__(labels=True), end='')

                if '.' in name:
                    printl(' [{}]'.format(name))
                else:
                    printl()
        else:
            printl('[none]')

@patchROOTClass
def GNAObject__variablevalues(self):
    ns = self.currentns
    return dict([(ns[k].name(), ns[k].value()) for k in self.variables.iterkeys()])
