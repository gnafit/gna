#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

trans = Variable('obj', 'i', 'j')( Variable('a', 'j', 'l')() )
trans2 = Variable('obj2', 'i', 'j')( Variable('b', 'n', 'o')() )
var1   = Variable('var1', 'j', 'k')
var2   = Variable('var2', 'k', 'm')

wt1 = var1*trans*var2*trans2
print(wt1, '=', wt1.estr(1))
print(wt1, '=', wt1.estr(2))
print(wt1, '=', wt1.estr())

print()
print('Walk+self')
for i, (l, o) in enumerate(wt1.walk(True)):
    print( i, l, '  '*l, o )

print()
print('Walk')
for i, (l, o) in enumerate(wt1.walk()):
    print( i, l, '  '*l, o )
