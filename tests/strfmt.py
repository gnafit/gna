#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna.labelfmt import reg_dictionary
from gna import labelfmt

mydict = dict(
        logL2='logL2 replaced',
        )
reg_dictionary( 'unfolding_corrected', mydict )

print( L('As is: {logL1_label}') )
print( L('Capitalize: {^logL1_label}') )
print( L('Indirect access to label: {$var}', var='logL1') )
print( L('Capitalize indirect: {^$var}', var='logL1_label') )

print( L('Unknown key: {unknown}') )

print( L('Overridden key: {logL2}') )

print(L('{^logL1_label}, {logL1}'))
print(L('{^logL2_label}, {logL2}'))

print(L('{0} {^logL2_label}', 'With positional args:'))

print('By short key: ', end='')
print(L.s('dm32'))
print()

print( 'Units and labels:' )
print('Name, unit, default offset:', L.u('dm32'))
print('Name, unit, custom offset:', L.u('dm32', offset=-5))
print('Name, unit, custom no offset:', L.u('dm32', offset=0))
print('Name, no unit:', L.u('theta13'))
print('Name, no unit, custom offset:', L.u('theta13', offset=-2))


