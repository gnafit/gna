#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna import labelfmt

labelfmt.labels['test'] = 'logL1'

print( L('As is: {test}') )
print( L('Capitalize: {^test}') )
print( L('Indirect access to label for test: {$test}') )

print( L('Unknown key: {unknown}') )

print(L('{^logL1_label}, {logL1}'))
print(L('{^logL2_label}, {logL2}'))

