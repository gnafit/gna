#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

print('Constructor')
print( Transformation( 'test' ) )
print( Transformation( 'test', 'i' ) )
print( Transformation( 'test', 'i', 'j' ) )

print( Transformation( 'test', '|', 'a' ) )
print( Transformation( 'test', 'i' , '|', 'a') )
print( Transformation( 'test', 'i', 'j', '|', 'a', 'b' ) )

print( Transformation( 'test', 'i', 'j', '|', Transformation('a', 'i', 'k'), 'b' ) )

print()

print('Call method')
t00 = Variable('t00')()
t10 = Variable('t10', 'i')()
t01 = Variable('t01')('a')
t11 = Variable('t11', 'i')('a')

print(t00)
print(t01)
print(t10)
print(t11)

