#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

eperfission = Transformation('eperf', 'i', 'r')
ffrac = Transformation('ffrac', 'i', 'r', 't')

eperfission_avg = ffrac*eperfission

print(eperfission)
print(ffrac)
print(eperfission_avg, '=', eperfission_avg.estr())

lib = dict([
    ('ffrac*eperf:i_r_t', dict(name='eperf')),
    ])

eperfission_avg.guessname(lib=lib, save=True)
print(eperfission_avg, '=', eperfission_avg.estr())
for i, (l, o, op) in enumerate(eperfission_avg.walk(True)):
    print( i, l, '  '*l, o, op )

