#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R

def run_unittests(glb, *args, **kwargs):
    message=kwargs.pop('message', 'All tests are OK!')
    for fcn in sorted([name for name in glb.keys() if name.startswith('test_')]):
        print('call ', fcn)
        glb[fcn](*args, **kwargs)
        print()

    print(message)

def passname(fcn):
    name = fcn.__name__
    def newfcn(*args, **kwargs):
        return fcn(*args, function_name=name, **kwargs)
    newfcn.__name__ = name
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
    def floatcopy(glb, addname=False):
        def decorator(fcn):
            if addname:
                return passname(fcn)
            else:
                return fcn
        return decorator

def addfcn(glb, name, addname=False):
    def adder(fcn):
        if addname:
            def newfcn(*args, **kwargs):
                return fcn(*args, function_name=name, **kwargs)
        else:
            newfcn=fcn
        newfcn.__name__ = name
        glb[name]=newfcn
        return newfcn

    return adder

def wrapfcn(glb, fcn, newname, addname, **contextkwargs):
    @addfcn(glb, newname, addname)
    def newfcn(*args, **kwargs):
        from gna import context
        with context.set_context(**contextkwargs) as cntx:
            return fcn(*args, **kwargs)

def clones(glb, float=False, gpu=False, npars=0, addname=False):
    precisions = [ 'double' ]
    gpus = [ False ]
    if float:
        precisions.append('float')
    if gpu:
        gpus.append(True)

    def decorator(fcn):
        for precision in precisions:
            for gpuon in gpus:
                if precision=='double' and not gpuon:
                    continue
                suffix = '_'+precision
                if gpuon:
                    suffix+='_gpu'
                newname = fcn.__name__+suffix

                wrapfcn(glb, fcn, newname, addname, precision=precision, gpu=gpuon, manager=npars)

        if addname:
            return passname(fcn)
        else:
            return fcn
    return decorator
