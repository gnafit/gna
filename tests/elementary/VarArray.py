#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env
import gna.constructors as C

ns = env.globalns
names = [ 'one', 'two', 'three', 'four', 'five' ]
for i, name in enumerate( names ):
    ns.defparameter( name, central=i, relsigma=0.1 )

vnames = C.stdvector( names )

va = R.VarArray( vnames )
output = va.vararray.points

ns.printparameters()
print( 'Array:', output.data() )


print('Change one input at a time:')
for i, name in enumerate( names, 2 ):
    ns[name].set(i)
    ns.printparameters()
    print( 'Array:', output.data() )
    print()


