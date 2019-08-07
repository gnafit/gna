#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.unittest import *

def test_passname():
    @passname
    def fcn(**kwargs):
        return kwargs.pop('function_name', None)

    assert fcn()=='fcn'

def test_addfcn():
    @addfcn(locals(), 'newfcn')
    def fcn(**kwargs):
        return kwargs.pop('function_name', None)

    glb = locals()
    assert fcn is glb['newfcn']

def test_clones_01():
    @clones(locals(), float=True, gpu=False, npars=1, addname=False)
    def fcn(**kwargs):
        pass

    glb=locals()
    assert glb['fcn']()==None
    assert glb['fcn_float']()==None

def test_clones_01():
    @clones(locals(), float=True, gpu=False, npars=1, addname=True)
    def fcn(**kwargs):
        return kwargs.pop('function_name', None)

    glb=locals()
    assert glb['fcn']()=='fcn'
    assert glb['fcn_float']()=='fcn_float'

def test_clones_02():
    @clones(locals(), float=True, gpu=True, npars=1, addname=True)
    def fcn(**kwargs):
        return kwargs.pop('function_name', None)

    glb=locals()
    assert glb['fcn']()=='fcn'
    # assert glb['fcn_double']()=='fcn_double'
    assert glb['fcn_float']()=='fcn_float'
    assert glb['fcn_double_gpu']()=='fcn_double_gpu'
    assert glb['fcn_float_gpu']()=='fcn_float_gpu'

def test_clones_contents():
    npars = 3

    @clones(locals(), float=True, gpu=True, npars=npars)
    def fcn(**kwargs):
        from gna.context import current_precision, current_precision_manager
        mainfcn = R.TransformationTypes.InitializerBase
        fcn = mainfcn.getDefaultFunction()
        precision_manager = current_precision_manager().current()
        if precision_manager:
            npars = precision_manager.getAllocator().maxSize()
        else:
            npars = None

        return current_precision(), fcn, npars

    glb = locals()
    assert glb['fcn']()            == ('double', '',    None)
    # assert glb['fcn_double']()     == ('double', '',    npars)
    assert glb['fcn_float']()      == ('float',  '',    npars)
    assert glb['fcn_double_gpu']() == ('double', 'gpu', npars)
    assert glb['fcn_float_gpu']()  == ('float',  'gpu', npars)

if __name__ == "__main__":
    run_unittests(globals())
