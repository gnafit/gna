#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna import labelfmt

print( L('As is: {logL1_label}') )
print( L('Capitalize: {^logL1_label}') )
print( L('Indirect access to label: {$var}', var='logL1') )
print( L('Capitalize indirect: {^$var}', var='logL1_label') )

print( L('Unknown key: {unknown}') )

print(L('{^logL1_label}, {logL1}'))
print(L('{^logL2_label}, {logL2}'))

