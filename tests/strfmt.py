#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.labelfmt import formatter as L
from gna.labelfmt import reg_dictionary
from gna import labelfmt

mydict = dict(
        logL2='logL2 replaced',
        dm32 = r'$\Delta m^2_{32}$',
        dm32_unit = r'$\text{eV}^2$',
        theta13 = r'$\sin^2 2\theta_13$',
        theta13_unit = '',
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

print(L.w_unit('dm32'))
print(L.w_unit('theta13'))

