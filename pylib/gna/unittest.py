#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R

def run_unittests(glb, message='All tests are OK!'):
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn]()
        print()

    print(message)

if 'float' in R.GNA.provided_precisions():
    def makefloat(name, globals):
        import constructors as C
        fcn=globals[name]
        def ffcn():
            with C.precision('float'):
                fcn()
        globals[name+'_float']=ffcn
else:
    def makefloat(name, globals):
        pass

