#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import configurator

cfg = configurator()
cfg.a=1
cfg['b']=2
cfg.set('c', 3)
print( cfg )
print()

cfg = configurator()
cfg('b')
cfg['b.d']=1
cfg.b.e=2
cfg.set('f.g',3)
print( cfg )
print()

cfg = configurator()
cfg('c.d.e').a=2
print( cfg )
print()

cfg = configurator()
cfg.a = dict(test=1, b=2)
cfg.b = { 1:2, 3:4 }
cfg('c.d.e').a= { 5:6 }
print( cfg )
print()
