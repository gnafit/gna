#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from gna.converters import convert

ns = env.globalns
names = [ 'one', 'two', 'three', 'four', 'five' ]
for i, name in enumerate( names ):
    ns.defparameter( name, central=(100-i*20), relsigma=0.1 )

vnames = convert( names, 'stdvector' )
vp = R.VarDiff( vnames, 'diff', ns=ns )
v=ns['diff'].get()
v.setLabel('a-b-c-...')

diff = ns['diff']

print_parameters(ns, labels=True)

print('Change one input at a time:')
for i, name in enumerate( names, 2 ):
    ns[name].set(i)
    print_parameters(ns)
    print()

print( 'Subtracto from 100' )
vp = R.VarDiff( vnames, 'diff100', 100., ns=ns )
v=ns['diff100'].get()
v.setLabel('100-a-b-c-...')
print_parameters(ns, labels=True)
