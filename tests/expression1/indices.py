#!/usr/bin/env python

from gna.expression import *

i0 = NIndex()
print('i0:', i0 )

i1 = NIndex( 'i' )
print('i1:', i1)

i2 = NIndex( 'i', 'j' )
print('i2:', i2)

j2 = NIndex( 'j', 'k' )
print('j2:', j2)

print('i1+i2:', i1+i2)

print('i2+j2:', i2+j2)

print('i2,m,n:', NIndex(i2, 'm', 'n'))

print('i1==i2:', i1==i2)
print('i2==i2:', i2==i2)
print('i1+i2==i2:', i1+i2==i2)

# print('reduce i2+j2 by k:', (i2+j2).reduce('k'))

print('ident', 'j2', j2.ident())
