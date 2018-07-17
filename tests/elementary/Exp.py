#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the Exp transformation"""

from __future__ import print_function
import numpy as N
from load import ROOT as R
from constructors import Points

arr = N.linspace( 0.0, 4.0, 81 )
print( 'Data:', arr )

"""Initialize transformations"""
points = Points( arr )
exp = R.Exp()
exp.exp.points(points.points)

data = exp.exp.result.data()
print( 'Result:', data )

cdata=N.exp(arr)
diff = data-cdata
ok = (N.fabs(diff)<1.e-12).all()
if not ok:
    print('Cmp result:', cdata)
    print('Diff:', diff)
print( ok and '\033[32mOK!' or '\033[31mFAIL!', '\033[0m' )

