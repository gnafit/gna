#!/usr/bin/env python

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

cat = Categories(OrderedDict(
        site=groups,
        exp ={ '': dets.keys() },
        det ={ d: (d,) for d in dets.keys() }
        ))

gd = GroupedDict( groups, sites )

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

print( 'Group keys' )
print( list(gd.groups.keys()) )
print()

print( 'Group keys (EH1)' )
print( list(gd.groups.keys('EH1')) )
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

print( 'Test categories:' )
print( 'AD11 in cat:', 'AD11' in cat )
print( 'EH11 in cat:', 'EH1' in cat )
print( 'AD11 group:', cat.group('AD11') )
print( 'AD11 group ("site"):', cat.group('AD11', 'site') )
print( 'AD11 group ("exp"):', cat.group('AD11', 'exp') or '""' )
print( 'AD11 groups:', cat.groups('AD11') )
print( 'AD11 categories:', cat.categories('AD11') )
print( 'AD11 items:', cat.itemdict('AD11') )
print( 'Format {exp}.{site}.{det}:', cat.format( 'AD11', '{exp}.{site}.{det}' ) )
print( 'Format {exp}.{site}.{det} (clean):', cat.format_splitjoin( 'AD11', '{exp}.{site}.{det}' ) )

cat.recursive=True
print( 'Test recursive categories:' )
print( 'AD11 in cat:', 'AD11' in cat )
print( 'AD11 group:', cat.group('AD11') )
print( 'AD11 groups:', cat.groups('AD11') )
print( 'AD11 cat:', cat.categories('AD11') )
print( 'AD11 items:', cat.itemdict('AD11') )
print( 'EH1 in cat:', 'EH1' in cat )
print( 'EH1 group:', cat.group('EH1') )
print( 'EH1 groups:', cat.groups('EH1') )
print( 'EH1 cat:', cat.categories('EH1') )
print( 'EH1 items:', cat.itemdict('EH1') )
