#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna import labelfmt

labelfmt.labels['test'] = 'logL1'

print( L('{^test}') ) # capitalize
print( L('{$test}') ) # indirect access

print(L('{^logL1_label}, {logL1}'))
print(L('{^logL2_label}, {logL2}'))

