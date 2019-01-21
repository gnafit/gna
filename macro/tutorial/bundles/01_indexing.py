#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.index import *

#
# 0d index
#
nidx = NIndex(fromlist=[])

print('Test 0d index')
for i, nit in enumerate(nidx):
    print('  iteration', i)
    print('    index:    ', nit.current_format() or '<empty string>')
    print('    full name:', nit.current_format(name='var'))
    print('    values:   ', nit.current_values())
    print()

#
# 1d index
#
nidx = NIndex(fromlist=[
    ('i', 'index', ['1', '2', '3'])
    ])

for i, nit in enumerate(nidx):
    print('  iteration', i, end=': ')
    print('    values:   ', nit.current_values())
print()


print('Test 1d index')
for i, nit in enumerate(nidx):
    print('  iteration', i, end=': ')
    print('    index:    ', nit.current_format())
print()

for i, nit in enumerate(nidx):
    print('  iteration', i, end=': ')
    print('    full name:', nit.current_format(name='var'))
print()

#
# 2d index
#
nidx = NIndex(fromlist=[
    ('i', 'index', ['1', '2', '3']),
    ('j', 'element', ['a', 'b'])
    ])

print('Test 2d index')
for i, nit in enumerate(nidx):
    print('  iteration', i)
    print('    index:    ', nit.current_format())
    print('    full name:', nit.current_format(name='var'))
    print('    values:   ', nit.current_values())

    print()

#
# 3d index and arbitrary name position
#
nidx = NIndex(fromlist=[
    ('z', 'clone', ['clone_00', 'clone_01']),
    'name',
    ('s', 'source', ['SA', 'SB']),
    ('d', 'detector', ['D1', 'D2'])
    ])
print('Test 3d index and arbitrary name position')
for i, nit in enumerate(nidx):
    print('  iteration', i)
    print('    full name:', nit.current_format(name='var'))
    print('    values:   ', nit.current_values(name='var'))

    print()

#
# 4d index and separated iteration
#
nidx = NIndex(fromlist=[
    ('z', 'clone', ['clone_00', 'clone_01']),
    'name',
    ('s', 'source', ['SA', 'SB']),
    ('d', 'detector', ['D1', 'D2']),
    ('e', 'element', ['e1', 'e2', 'e3'])
    ])
print('Test 4d index and separated iteration')
nidx_major, nidx_minor=nidx.split(('s', 'd'))
for i_major, nit_major in enumerate(nidx_major):
    print('  major iteration', i_major)
    print('    major values:   ', nit_major.current_values())

    for j_minor, nit_minor in enumerate(nidx_minor):
        print('      minor iteration', j_minor)
        print('        minor values:  ', nit_minor.current_values())

        nit = nit_major + nit_minor
        print('        full name:     ', nit.current_format(name='var'))
        print('        custom label:  ', nit.current_format('Flux from {source} to {detector} element {element} ({clone})'))

    print()
    break

#
# Dependant indices
#
nidx = NIndex(fromlist=[
    ('d', 'detector', ['D1', 'D2']),
    ('s', 'source', ['SA', 'SB']),
    ('g', 'group', ['g1', 'g2']),
    ('e', 'element', ['e1', 'e2', 'e3'], dict(short='g', name='group', map=[('g1', ('e1', 'e2')), ('g2', ('e3',)) ]))
    ])

print('Test 4d index and dependant indices')
nidx_major, nidx_minor=nidx.split(('d', 'g'))
nidx_e=nidx.get_subset('e')
for i_major, nit_major in enumerate(nidx_major):
    print('  major iteration', i_major)
    print('    major values:   ', nit_major.current_values())

    for j_minor, nit_minor in enumerate(nidx_minor):
        nit = nit_major + nit_minor
        print('      full values %i:'%j_minor, nit.current_values())

    print()

print('Test 4d index and separated iteration: try to mix dependent indices')
nidx_major+=nidx_e
for i_major, nit_major in enumerate(nidx_major):
    print('  major iteration', i_major)
    print('    major values:   ', nit_major.current_values())

    for j_minor, nit_minor in enumerate(nidx_minor):
        nit = nit_major + nit_minor
        print('      full values %i:     '%j_minor, nit.current_values())
        print('      formatted string %i:'%j_minor, nit.current_format('Element {element} in group {group}'))

    print()
