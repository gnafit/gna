#!/usr/bin/env python

from gna.configurator import configurator
from pprint import pprint

print( 'Cfg 1' )
c = configurator( 'tests/elementary/test_cfgloader_cfg1.py', debug=True )
pprint( c )
print()

print( 'Cfg 2' )
c.__load__(  'tests/elementary/test_cfgloader_cfg2.py' )
pprint( c )
print()

print( 'Cfg 3 (nested)' )
c.__load__(  'tests/elementary/test_cfgloader_cfg3.py' )
pprint( c )
print()

print( 'Test subst (list)' )
c = configurator( 'tests/elementary/test_cfgloader_{0}.py', debug=True, subst=['cfg1', 'cfg2'] )
pprint( c )
print()

print( 'Test subst (dict)' )
c = configurator( 'tests/elementary/test_cfgloader_{location}.py', debug=True, subst=dict(key='location', values=['cfg1', 'cfg2', 'cfg3']) )
pprint( c )
