#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from gna import constructors as C

ns = env.globalns
names = [ 'one', 'two', 'three', 'four', 'five' ]
for i, name in enumerate( names, 1 ):
    ns.defparameter( name, central=i, relsigma=0.1 )

vp = C.VarSum(names, 'sum', ns=ns)
ns['sum'].get()

sum = ns['sum']

print_parameters(ns)

print('Change one input at a time:')
for i, name in enumerate( names, 2 ):
    ns[name].set(i)
    print_parameters(ns)
    print()


