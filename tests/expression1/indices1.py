#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

indices = [
        ( 'r', 'reactor', ('DB', 'LA1', 'LA2') ),
        ( 'd', 'detector', ('AD11', 'AD12', 'AD21') ),
        ( 'c', 'component', ('comp0', 'comp1', 'comp2') )
        ]

def rules( nidx ):
    if nidx.indices['c'].current == 'comp0':
        nidx.indices['r'].current = 'all'
        nidx.indices['d'].current = 'all'
    return nidx

nidx = NIndex( fromlist=indices, rules=rules )
print('nidx:', nidx )

for it in nidx:
    print( it.current_format() )

