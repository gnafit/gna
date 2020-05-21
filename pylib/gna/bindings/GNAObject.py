# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from gna.bindings import patchROOTClass, DataType, provided_precisions
from gna.bindings import DataType
import ROOT as R
from printing import printl, nextlevel
import numpy as np

classes = []
classes_single=[]
for ft in provided_precisions:
    classes.append(R.GNAObjectT(ft, ft))
    classes_single.append(R.GNASingleObjectT(ft, ft))
    classes_single.append(R.SingleOutputT(ft))

@patchROOTClass(classes+classes_single, '__getattr__')
def GNAObject____getattr__(self, attr):
    try:
        return self[attr]
    except KeyError:
        raise AttributeError(attr)

@patchROOTClass(classes, '__str__')
def GNAObject____str__(self):
    return '[obj] {}: {:d} transformation(s), {:d} variables'.format(self.__class__.__name__, self.transformations.size(), self.variables.size())

@patchROOTClass(classes, 'print')
def GNAObject__print(self, **kwargs):
    print_trans = kwargs.pop('transformations', True)
    print_vars = kwargs.pop('variables', True)

    if print_vars:
        self.printtransformations(**kwargs)

    if not self.variables or not print_vars:
        return

    if self.variables.size()>0:
        print()
        self.printvariables()

@patchROOTClass(classes, 'printtransformations')
def GNAObject__printtransformations(self, **kwargs):
    printl(str(self))
    if self.transformations.size():
        with nextlevel():
            # printl('Transformations:')
            for i, t in enumerate(self.transformations.itervalues()):
                printl('{:2d}'.format(i), end=' ')
                t.print(**kwargs)

@patchROOTClass(classes, 'printvariables')
def GNAObject__printvariables(self):
    from gna.parameters import printer
    ns = self.currentns

    printl('Variables, relative to namespace "{}":'.format(ns.name))
    with nextlevel():
        if self.variables.size()>0:
            for name in self.variables:
                if name in ns:
                    printl(ns[name].__str__(labels=True), end='')

                    if '.' in name:
                        printl(' [{}]'.format(name))
                    else:
                        printl()
                else:
                    printl('unknown', name)
        else:
            printl('[none]')

@patchROOTClass(classes, 'variablevalues')
def GNAObject__variablevalues(self):
    ns = self.currentns
    return dict([(ns[k].name(), ns[k].value()) if k in ns else (k+'_unknown', 0) for k in self.variables.iterkeys()])

R.SingleOutput.__single_orig = R.SingleOutput.single
@patchROOTClass(classes, 'single')
def SingleOutput__single(self):
    R.OutputDescriptor(self.__single_orig())

@patchROOTClass(classes_single, 'data')
def GNAObject__data(self):
    buf = self.single().__data_orig()
    datatype = self.datatype()
    return np.frombuffer(buf, count=datatype.size(), dtype=buf.typecode).reshape(datatype.shape, order='F')

@patchROOTClass(classes+classes_single, 'single')
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
def GNAObject____rlshift__(obj, inputs):
    '''inputs(argument)<<outputs(self)'''
    obj.single()>>inputs

@patchROOTClass(classes, '__lshift__')
def GNAObject____lshift__(obj, output):
    '''inputs(obj)<<output(arg)'''
    output>>obj.single_input()

@patchROOTClass(classes, ('__gt__', '__lt__'))
def GNAObject______cmp__(a, b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
