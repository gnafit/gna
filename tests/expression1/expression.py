#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

indices = [
        ('i', 'index_i', '1', '2', '3'),
        ('j', 'index_j', '4', '5', '6'),
        ('b', 'index_b', 'a', 'b', 'c')
        ]
lib = dict(
        [
            ('var*obj', dict(name='weighted')),
            ('weighted+bkg', dict(name='signal'))
            ]
        )
expr = 'mat[b]|fun|var[j]*obj[i]()+bkg()'
a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.tree.guessname( lib=lib, save=True )
a.tree.dump(True)

