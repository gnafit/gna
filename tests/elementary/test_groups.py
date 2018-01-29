#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.grouping import *
from collections import OrderedDict

dets = OrderedDict( [
    ('AD11', 1), ('AD12', 2),
    ('AD21', 3), ('AD22', 8),
    ('AD31', 4), ('AD32', 5),
    ('AD33', 6), ('AD33', 7)
    ] )

sites = OrderedDict( [
    ('EH1', 'A'),
    ('EH2', 'B'),
    ('EH3', 'C')
    ] )

groups = OrderedDict( [
    ( 'EH1', ('AD11', 'AD12') ),
    ( 'EH2', ('AD21', 'AD22') ),
    ( 'EH3', ('AD31', 'AD32', 'AD33', 'AD34') ),
    ] )

groupings = GroupsSet(OrderedDict(
        site=groups,
        exp ={ '': dets.keys() },
        det ={ d: (d,) for d in dets.keys() }
        ))

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

print()

print( 'Test sets' )
print( '11 in sets:', 'AD11' in groupings )
print( '11 group:', groupings.group('AD11') )
print( '11 group ("site"):', groupings.group('AD11', 'site') )
print( '11 group ("exp"):', groupings.group('AD11', 'exp') )
print( '11 groups:', groupings.groups('AD11') )
print( '11 items:', groupings.items('AD11') )
print( 'Format {exp}.{site}.{det}:', groupings.format( 'AD11', '{exp}.{site}.{det}' ) )
print( 'Format {exp}.{site}.{det} (clean):', groupings.format_splitjoin( 'AD11', '{exp}.{site}.{det}' ) )
