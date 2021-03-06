#!/usr/bin/env python

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
eperfission_avg.dump(True)

