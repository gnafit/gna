#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import configurator
from pprint import pprint

print( 'Cfg 1' )
c = configurator( 'tests/elementary/cfgloader_cfg1.py', debug=True )
pprint( c )
print()

print( 'Cfg 2' )
c.__load__(  'tests/elementary/cfgloader_cfg2.py' )
pprint( c )
print()

print( 'Cfg 3 (nested)' )
c.__load__(  'tests/elementary/cfgloader_cfg3.py' )
pprint( c )
print()

print( 'Test subst (list)' )
c = configurator( 'tests/elementary/cfgloader_{0}.py', debug=True, subst=['cfg1', 'cfg2'] )
pprint( c )
print()

print( 'Test subst (dict)' )
c = configurator( 'tests/elementary/cfgloader_{location}.py', debug=True, subst=dict(key='location', values=['cfg1', 'cfg2', 'cfg3']) )
pprint( c )
