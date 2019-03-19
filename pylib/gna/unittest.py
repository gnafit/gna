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

def passname(fcn):
    name = fcn.__name__
    def newfcn(*args, **kwargs):
        return fcn(*args, function_name=name, **kwargs)
    return newfcn

if 'float' in R.GNA.provided_precisions():
    def floatcopy(glb, addname=False):
        def decorator(fcn):
            newname = fcn.__name__+'_float'
            def newfcn(*args, **kwargs):
                from gna import context
                if addname:
                    kwargs.setdefault('function_name', newname)
                with context.precision('float'):
                    fcn(*args, **kwargs)
            newfcn.__name__=newname
            glb[newname]=newfcn

            if addname:
                return passname(fcn)
            else:
                return fcn
        return decorator
else:
    def floatcopy(*args, **kwargs):
        return lambda fcn: fcn
