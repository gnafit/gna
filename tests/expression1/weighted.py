#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

trans = Variable('obj', 'i', 'j')( Variable('a', 'j', 'l')() )
trans2 = Variable('obj2', 'i', 'j')( Variable('b', 'n', 'o')() )
var1   = Variable('var1', 'j', 'k')
var2   = Variable('var2', 'k', 'm')
wt = var1*trans
wt.name = 'weighted1'
wt1 = trans*var1
wt1.name = 'weighted2'
print( var1 )
print( trans )
print(wt, '=', wt.estr())
print(wt1, '=', wt.estr())

print('More variables:')
wt1 = wt1*var2
wt1.name='weighted1'
wt1.weight.name='w1'
print(wt1, '=', wt1.estr(1))
print(wt1, '=', wt1.estr(2))
print(wt1, '=', wt1.estr())

print('More transformations:')
wt1 = wt1*trans2
wt1.name='weighted1'
wt1.weight.name='w1'
wt1.object.name='o1'
print(wt1, '=', wt1.estr(1))
print(wt1, '=', wt1.estr(2))
print(wt1, '=', wt1.estr())
