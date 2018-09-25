#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *


print('Call method')
t00 = Variable('t00')()
t10 = Variable('t10', 'i')()
t01 = Variable('t01')('a')
t11 = Variable('t11', 'j')('a')

print(t00)
print(t01)
print(t10)
print(t11)

print()
print('Product')
tp = t10*t11
tp.name = 'tprod'
print( tp, '=', tp.estr() )

print()
print('Sum')
ts = t10+t11
ts.name = 'sum'
print( ts, '=', ts.estr() )

print()
print('Sum and product')
ts2 = ts+ts
ts2.name = 'sumsum'
print( ts2, '=', ts2.estr() )

tp2 = tp+tp
tp2.name = 'sumprod'
print( tp2, '=', tp2.estr() )

tp2 = tp*tp
tp2.name = 'prodprod'
print( tp2, '=', tp2.estr() )

tp2 = ts*ts
tp2.name = 'prodsum'
print( tp2, '=', tp2.estr() )

print()
print('ident', 'tp2', tp2.ident())
print('ident full', 'tp2', tp2.ident_full())
