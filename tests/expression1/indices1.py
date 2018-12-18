#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

indices = [
        ( 'z', 'clone', ('clone00', 'clone01', 'clone02') ),
        ( 'd', 'detector', ('AD11', 'AD12', 'AD21') ),
        ( 'r', 'reactor', ('DB', 'LA1', 'LA2') ),
        ( 'c', 'component', ('comp0', 'comp1', 'comp2') ),
        ]

nidx = NIndex(fromlist=indices, name_position=2)
print('nidx:', nidx)
print('order:', nidx.order)

for it in nidx:
    print(it.current_format(name='fake'))

n1, n2 = nidx.split(('z', 'r'))

print()
print('Iterate n1')
for it in n1:
    print(it.current_format(name='fake'))

print()
print('Iterate n2')
for it in n2:
    print(it.current_format(name='fake'))


n3 = nidx.get_subset(('z', 'd'))
print()
print('Iterate n3')
for it in n3:
    print(it.current_format(name='fake'))

n4 = nidx.get_subset([])
print()
print('Iterate n4')
for it in n4:
    print(it.current_format(name='fake'))

n5 = nidx.get_subset(['d', 'c'])
print()
print('Iterate n5')
for it in n5:
    print(it.current_format(name='fake'))

n6 = nidx.get_subset(['r', 'c'])
print()
print('Iterate n6')
for it in n6:
    print(it.current_format(name='fake'))
