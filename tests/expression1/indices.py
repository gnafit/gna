#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

i0 = Indices()
print('i0:', i0 )

i1 = Indices( 'i' )
print('i1:', i1)

i2 = Indices( 'i', 'j' )
print('i2:', i2)

j2 = Indices( 'j', 'k' )
print('j2:', j2)

print('i1+i2:', i1+i2)

print('i2+j2:', i2+j2)

print('i2,m,n:', Indices(i2, 'm', 'n'))

print('i1==i2:', i1==i2)
print('i2==i2:', i2==i2)
print('i1+i2==i2:', i1+i2==i2)

print('reduce i2+j2 by k:', (i2+j2).reduce('k'))
