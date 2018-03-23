#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
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

print('reduce', op, 'by k', op.reduce('op_reduced', 'k'))
