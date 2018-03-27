#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

trans = Variable('obj1', 'i', 'j')( Variable('a', 'j', 'l')() )
trans2 = Variable('obj2', 'i', 'j')( Variable('b', 'n', 'o')() )
trans3 = Variable('obj3', 'j', 'l')()
var1   = Variable('var1', 'j', 'k')
var2   = Variable('var2', 'k', 'm')

wt1 = var1*trans*var2*(trans2+trans3)
print(wt1, '=', wt1.estr(1))
print(wt1, '=', wt1.estr(2))
print(wt1, '=', wt1.estr())

print()
print('Raw calculation tree state')
print('Walk+self')
for i, (l, o, op) in enumerate(wt1.walk(True)):
    print( i, l, '  '*l, o, op )

print()
print('Walk')
for i, (l, o, op) in enumerate(wt1.walk()):
    print( i, l, '  '*l, o, op )

print()
print('ident', 'wt1', wt1.ident())
print('ident full', 'wt1', wt1.ident_full())

lib = dict([
    ('var1*var2:j_k_m', dict(name='prod12')),
    ('obj2+obj3',       dict(name='sum23') ),
    ('obj1*sum23',      dict(name='os123') ),
    ('os123*prod12',    dict(name='final') ),
    ])
print()
print('Guess names from:')
print(lib)
wt1.guessname(lib=lib, save=True)

print(wt1, '=', wt1.estr(1))
print(wt1, '=', wt1.estr(2))
print(wt1, '=', wt1.estr())

print()
print('Walk+self')
for i, (l, o, op) in enumerate(wt1.walk(True)):
    print( i, l, '  '*l, o, op )

print()
print('Walk')
for i, (l, o, op) in enumerate(wt1.walk()):
    print( i, l, '  '*l, o, op )

print()
print('ident', 'wt1', wt1.ident())
print('ident full', 'wt1', wt1.ident_full())
