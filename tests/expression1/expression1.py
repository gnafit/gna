#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *
from gna.bundle import execute_bundle

indices = [
    ('n', 'num',   ['1', '2', '3']),
    ('a', 'alph',  ['a', 'b', 'c']),
    ('z', 'zyx',   ['X', 'Y', 'Z'])
    ]

lib = dict(
        [
            ('norm*spec', dict(name='obs_spec')),
            ]
        )

expr = 'norm()*spec[n]()'
a = Expression(expr, indices=indices)

print(a.expression_raw)
print(a.expression)

a.parse()
a.guessname(lib=lib, save=True)
a.tree.dump(True)

print()
cfg = NestedDict(
        norm = NestedDict(
            bundle = 'dummy',
            format = 'norm.{autoindex}',
            input = True,
            size = 10,
            debug = False
            ),
        spec = NestedDict(
            bundle = 'dummy',
            format = 'spec.{autoindex}',
            input = True,
            size = 10,
            debug = True
            )
        )
context = ExpressionContext( cfg )
a.build(context)
