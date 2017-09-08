#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from string import Formatter

class LFormatter(Formatter):
    def get_value( self, key, args, kwargs ):
        if type(key) is long:
            return args[key]

        if key in kwargs:
            return kwargs[key]

        if key.startswith( '__' ):
            return kwargs[key[2:]].capitalize()

        return key

lfmt = LFormatter()

labels = dict(
    logL1_label = 'agreement to data',
    logL1       = r'$\log L_1 = \log \chi^2$',
    logL2_label = 'regularity',
    logL2       = r'$\log L_2/\tau$'
)
def L( s ):
    return lfmt.format( s, **labels )

print(L('{__logL1_label}, {logL1}'))
print(L('{__logL2_label}, {logL2}'))
