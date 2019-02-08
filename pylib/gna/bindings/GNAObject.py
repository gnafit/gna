# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.bindings import patchROOTClass, DataType, provided_precisions
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel

classes = [R.GNAObjectT(ft,ft) for ft in provided_precisions]

@patchROOTClass(classes, '__str__')
def GNAObject____str__(self):
    return '[obj] {}: {:d} transformation(s), {:d} variables'.format(self.__class__.__name__, self.transformations.size(), self.variables.size())

@patchROOTClass(classes, 'print')
def GNAObject__print(self):
    self.printtransformations()
    if not self.variables:
        return

    if self.variables.size()>0:
        print()
        self.printvariables()

@patchROOTClass(classes, 'printtransformations')
def GNAObject__printtransformations(self):
    printl(str(self))
    if self.transformations.size():
        with nextlevel():
            # printl('Transformations:')
            for i, t in enumerate(self.transformations.itervalues()):
                printl('{:2d}'.format(i), end=' ')
                t.print()

@patchROOTClass(classes, 'printvariables')
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

@patchROOTClass(classes, 'variablevalues')
def GNAObject__variablevalues(self):
    ns = self.currentns
    return dict([(ns[k].name(), ns[k].value()) for k in self.variables.iterkeys()])

R.SingleOutput.__single_orig = R.SingleOutput.single
@patchROOTClass(classes, 'single')
def SingleOutput__single(self):
    R.OutputDescriptor(self.__single_orig())

@patchROOTClass(classes+[R.GNASingleObject, R.SingleOutput], 'single')
def GNAObject__single(self):
    transf = self.transformations
    if transf.size()!=1:
        raise Exception('Can not call single() on object with %i transformations'%(self.transformations.size()))

    return transf.front().single()

@patchROOTClass(classes, 'single_input')
def GNAObject__single_input(self):
    transf = self.transformations
    if transf.size()!=1:
        raise Exception('Can not call single_input() on object with %i transformations'%(self.transformations.size()))

    return transf.front().single_input()

@patchROOTClass(classes, '__rshift__')
def GNAObject______rshift__(obj, inputs):
    '''output(self)>>inputs(arg)'''
    obj.single()>>inputs


@patchROOTClass(classes, '__rlshift__')
def GNAObject______rlshift__(obj, inputs):
    '''inputs(argument)<<outputs(self)'''
    obj.single()>>inputs

@patchROOTClass(classes, '__lshift__')
def GNAObject______lshift__(obj, output):
    '''inputs(obj)<<output(arg)'''
    output>>obj.single_input()

@patchROOTClass(classes, ('__gt__', '__lt__'))
def GNAObject______cmp__(a,b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
