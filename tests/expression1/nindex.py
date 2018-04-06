#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

idx1 = Index('i', 'Idx', ('a', 'b', 'c', 'd'))
jdx2 = Index('j', 'Jdx', ('1', '2', '3'))
kdx3 = Index('k', 'Kdx', ('x', 'y'))

ndx = NIndex( idx1, jdx2, kdx3 )
print( idx1 )
print( jdx2 )
print( kdx3 )
print( ndx )

print( 'Iterate i' )
for i in idx1:
    print( *i )
print()

print( 'Iterate i (fixed)' )
for i in idx1.iterate(fix=dict(i='b')):
    print( i )
print()

print( 'Iterate nidx' )
for i in ndx:
    print( *i )
print()

print( 'Iterate nidx: fix i' )
for i in ndx.iterate(fix=dict(i='b')):
    print( *i )
print()

print( 'Iterate nidx: fix j' )
for i in ndx.iterate(fix=dict(j='2')):
    print( *i )
print()

print( 'Iterate nidx: fix k, j' )
for i in ndx.iterate(fix=dict(k='y', j='2')):
    print( *i )
print()

print( 'Iterate nidx: fix k, j, i' )
for i in ndx.iterate(fix=dict(k='y', j='2', i='a')):
    print( *i )
print()

print( 'Iterate nidx (items): fix k, j' )
for i in ndx.iterate(mode='items', fix=dict(k='y', j='2')):
    print( *i )
print()

print( 'Iterate nidx (format): fix k, j' )
for i in ndx.iterate(mode='i={i}, j={j}, k={k}', fix=dict(k='y', j='2')):
    print( i )
print()

print( 'Iterate nidx (format): fix k, j' )
for i in ndx.iterate(mode='j={j}, k={k}, i={i}', fix=dict(k='y', j='2')):
    print( i )
print()
