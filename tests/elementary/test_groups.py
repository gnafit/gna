#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.grouping import *
from collections import OrderedDict

dets = OrderedDict( [
    ('11', 1), ('12', 2),
    ('21', 3), ('22', 8),
    ('31', 4), ('32', 5),
    ('33', 6), ('33', 7)
    ] )

sites = OrderedDict( [
    ('EH1', 'A'),
    ('EH2', 'B'),
    ('EH3', 'C')
    ] )

groups = OrderedDict( [
    ( 'EH1', ('11', '12') ),
    ( 'EH2', ('21', '22') ),
    ( 'EH3', ('31', '32', '33', '34') ),
    ] )

gd = GroupedDict( sites, groups=groups )

print( 'Retrieve data by actual keys (sites):' )
for k, v in gd.items():
    print( '  ', k, v )
print()

print( 'Retrieve data by detectors' )
for k, v in gd.subitems():
    print( '  ', k, v )
print()

print( 'Keys' )
print( list(gd.keys()) )
print()

print( 'Sub keys' )
print( list(gd.subkeys()) )
print()

print( 'Items' )
print( list(gd.items()) )
print()

print( 'Sub items' )
print( list(gd.subitems()) )
print()

print( 'Values' )
print( list(gd.values()) )
print()

print( 'Sub values' )
print( list(gd.subvalues()) )
print()

print( 'Retrieve missing key (KeyError): ', end='')
try:
    gd['missing']
except KeyError:
    print( '\033[32mOK!\033[0m' )
except:
    print( '\033[32mFail!\033[0m' )

