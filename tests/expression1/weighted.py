#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression import *

trans = Variable('obj', 'i', 'j')('a')
var   = Variable('var', 'j', 'k')

wt = var*trans
wt.name = 'weighted1'
wt1 = trans*var
wt1.name = 'weighted2'
print( var )
print( trans )
print(wt, '=', wt.estr())
print(wt1, '=', wt.estr())

