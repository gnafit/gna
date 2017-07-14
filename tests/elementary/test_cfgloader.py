#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.cfgloader import config
from pprint import pprint

print( 'Cfg 1' )
c = config( 'tests/elementary/test_cfgloader_cfg1.py', debug=True )
pprint( c.__dict__ )
print()

print( 'Cfg 2' )
c.__load__(  'tests/elementary/test_cfgloader_cfg2.py' )
pprint( c.__dict__ )

