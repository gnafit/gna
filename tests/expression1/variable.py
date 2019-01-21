#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

v0 = Variable('v0')
print( v0 )

v1 = Variable('v1', 'i')
print(v1)

v2 = Variable('v2', 'i', 'j')
print(v2)

v3 = Variable('v3', 'j', 'k')
print(v3)

v4 = Variable('v4', 'm')
print(v4)

pr1 = v2*v3
pr1.name = 'pr1'
print( 'pr1=v2*v3:', pr1, '=', pr1.estr() )

pr2 = pr1*v4
pr2.name = 'pr2'
print( 'pr2=pr1*v4:', pr2, '=', pr2.estr() )

pr3 = v4*pr1
pr3.name = 'pr3'
print( 'pr3=v4*pr1:', pr3, '=', pr3.estr() )


w1 = Variable('w1', 'k')
print(w1)
w2 = Variable('w2', 'l', 'm')
print(w2)
w3 = Variable('w3', 'm', 'n')
print(v3)

pr4=w1*w2*w3
pr4.name='pr4'
print('pr4:', pr4, '=', pr4.estr())

pr5 = pr3*pr4
pr5.name = 'pr5'
print( 'pr5=pr3*pr4:', pr5, '=', pr5.estr() )

pr6 = pr4*pr3
pr6.name = 'pr6'
print( 'pr6=pr4*pr3:', pr6, '=', pr6.estr() )

print()
pr6.name='?'
print('ident', 'pr6', pr6.ident())
print('ident full', 'pr6', pr6.ident_full())
