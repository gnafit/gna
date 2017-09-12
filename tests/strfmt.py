#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna.labelfmt import reg_dictionary
from gna import labelfmt

mydict = dict(
        mylabel = 'my label',
        anotherlabel = 'another label'
        )
reg_dictionary( 'mydict', mydict )

print("Printing labels")
print( L('  my label: {mylabel}') )
print( L('  two labels: {mylabel} and {anotherlabel}') )
print( L('  unknown key representation: {unknown}') )
print( L('  {1} {0} arguments: {mylabel}', 'positional', 'using'))
print()

print("Simple acecss to labels")
print('  by short key:', L.s('mylabel'))
print('  by short key:', L.base('mylabel'))
print()

print("Simple modifications")
print( L('  capitalize: {^mylabel}') )
print( L('  indirect access to label from key "var": {$var}', var='mylabel') )
print( L('  capitalize indirect: {^$var}', var='mylabel') )
print()

print( 'Units and labels:' )
print('  name, unit, default offset:', L.u('dm32'))
print('  name, unit, custom offset:', L.u('dm32', offset=-5))
print('  name, unit, force no offset:', L.u('dm32', offset=0))
print()

print('  name, no unit:', L.u('theta13'))
print('  name, no unit, custom offset:', L.u('theta13', offset=-2))


