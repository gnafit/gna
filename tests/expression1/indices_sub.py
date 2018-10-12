#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from collections import OrderedDict

idx = Index('t', 'test', ('a', 'b', 'c', 'd'), sub=dict(short='s', name='sub', map=OrderedDict(A=('a'), B=('b', 'c', 'd'))))
idx2 = Index('o', 'other', ('1', '2', '3', '4'))
idx_sub = idx.sub
print('idx:', idx )

print('single')
print(str(idx))
for it in idx.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in idx.iterate():
    print('  ', it.current_items(include_sub=False))
print()

print('sub single')
print(str(idx_sub))
for it in idx_sub.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in idx_sub.iterate():
    print('  ', it.current_items(include_sub=False))
print()

print('main')
nidx1 = NIndex(idx, idx2)
print(str(nidx1))
for it in nidx1.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in nidx1.iterate():
    print('  ', it.current_items(include_sub=False))
print()

print('sub')
nidx2 = NIndex(idx2, idx_sub)
print(str(nidx2))
for it in nidx2.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in nidx2.iterate():
    print('  ', it.current_items(include_sub=False))
print()

print('overridden')
nidx3 = NIndex(idx, idx2, idx_sub)
print(str(nidx3))
for it in nidx3.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in nidx3.iterate():
    print('  ', it.current_items(include_sub=False))
print()

print('overriding')
nidx4 = NIndex(idx_sub, idx2, idx)
print(str(nidx4))
for it in nidx4.iterate():
    print('  ', it.current_items(include_sub=True))
print('again')
for it in nidx4.iterate():
    print('  ', it.current_items(include_sub=False))
print()

var = Variable('var', idx)
trans = Transformation('trans', idx_sub)
from load import ROOT as R
prod = var*trans

prod.test_iteration()

