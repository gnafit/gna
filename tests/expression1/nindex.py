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
    print( '   ', i.name, i.current )
print()

print( 'Iterate i (fixed)' )
for i in idx1.iterate(fix=dict(i='b')):
    print( '   ', i.name,  i.current )
print()

print( 'Iterate nidx' )
for i, it in enumerate(ndx):
    print( i, it.current_values() )
print()

print( 'Iterate nidx (short)' )
for i, it in enumerate(ndx):
    print( i, it.current_items( 'short' ) )
print()

print( 'Iterate nidx (long)' )
for i, it in enumerate(ndx):
    print( i, it.current_items( 'long' ) )
print()

print( 'Iterate nidx (both)' )
for i, it in enumerate(ndx):
    print( i, it.current_items( 'both' ) )
print()

print( 'Iterate nidx (fmt)' )
for i, it in enumerate(ndx):
    print( i, it.current_format('i={Idx}, j={j}, k={Kdx}, extra={extra}', extra='test') )
print()

print( 'Iterate nidx (autofmt)' )
for i, it in enumerate(ndx):
    print( i, it.current_format() )
print()

print( 'Iterate nidx (semi autofmt)' )
for i, it in enumerate(ndx):
    print( i, it.current_format( '{name}.{autoindex}', name='testname' ) )
print()


# print( 'Iterate nidx: fix i' )
# for i in ndx.iterate(fix=dict(i='b')):
    # print( *i )
# print()

# print( 'Iterate nidx: fix j' )
# for i in ndx.iterate(fix=dict(j='2')):
    # print( *i )
# print()

# print( 'Iterate nidx: fix k, j' )
# for i in ndx.iterate(fix=dict(k='y', j='2')):
    # print( *i )
# print()

# print( 'Iterate nidx: fix k, j, i' )
# for i in ndx.iterate(fix=dict(k='y', j='2', i='a')):
    # print( *i )
# print()

# print( 'Iterate nidx (items): fix k, j' )
# for i in ndx.iterate(mode='items', fix=dict(k='y', j='2')):
    # print( *i )
# print()

# print( 'Iterate nidx (format): fix k, j' )
# for i in ndx.iterate(mode='i={i}, j={j}, k={k}', fix=dict(k='y', j='2')):
    # print( i )
# print()

# print( 'Iterate nidx (format): fix k, j' )
# for i in ndx.iterate(mode='j={j}, k={k}, i={i}', fix=dict(k='y', j='2')):
    # print( i )
# print()
