#!/usr/bin/env python

from gna.expression import *

i0 = Indexed('i0')
print( i0 )

i1 = Indexed('i1', 'i')
print(i1)

i2 = Indexed('i2', 'i', 'j')
print(i2)

j2 = Indexed('j2', 'j', 'k')
print(j2)

op = Indexed('operation_i1_i2_j2', i1, i2, j2)
print( op )

print( 'i1==i2', i1==i2 )
print( 'i1==i1', i1==i1 )

# print('reduce', op, 'by k', op.reduce('op_reduced', 'k'))

print('ident', 'j2', j2.ident())
print('ident full', 'j2', j2.ident_full())

print()
print('Testing __getitem__ syntax')
b = Index('b', 'longindex', ['b1', 'b1'])
v = Indexed('v')
print( v )
print( v['a', b] )
print( v )
try:
    v['a', b]
except:
    print( 'OK' )
