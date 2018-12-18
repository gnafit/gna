#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the SelfPower transformation"""

from __future__ import print_function
from matplotlib import pyplot as P
import numpy as N
from load import ROOT as R
from gna.constructors import Points
from gna.env import env

"""Initialize parameters and data"""
ns = env.globalns
varname = 'scale'
par=ns.defparameter( varname, central=1.0, sigma=0.1 )

arr = N.linspace( 0.0, 4.0, 81 )
print( 'Data:', arr )

"""Initialize transformations"""
points = Points( arr )
selfpower = R.SelfPower( varname )

selfpower.selfpower.points(points.points)
selfpower.selfpower_inv.points(points.points)

checks = ()

"""
Plot results
(Plot for positive power)
"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '$x$' )
ax.set_ylabel( 'f(x)' )
ax.set_title( 'SelfPower: $(x/a)^{x/a}$' )

"""a=1"""
par.set(1)
data = selfpower.selfpower.result.data().copy()
ax.plot( arr, data, label='$a=1$' )

checks += data - (arr/par.value())**(arr/par.value()),

"""a=2"""
par.set(2)
data = selfpower.selfpower.result.data().copy()
ax.plot( arr, data, label='$a=2$' )

checks += data - (arr/par.value())**(arr/par.value()),

ax.legend(loc='upper left')

"""
Plot for negative power
"""
fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '$x$' )
ax.set_ylabel( 'f(x)' )
ax.set_title( 'SelfPower: $(x/a)^{-x/a}$' )

"""a=1"""
par.set(1)
data = selfpower.selfpower_inv.result.data().copy()
ax.plot( arr, data, label='$a=1$' )

checks += data - (arr/par.value())**(-arr/par.value()),

"""a=2"""
par.set(2)
data = selfpower.selfpower_inv.result.data().copy()
ax.plot( arr, data, label='$a=2$' )

checks += data - (arr/par.value())**(-arr/par.value()),

ax.legend(loc='upper right')

"""Cross check results with numpy calculation"""
checks = N.array( checks )
print( (checks==0.0).all() and '\033[32mCross checks passed OK!' or '\033[31mCross checks failed!', '\033[0m' )

P.show()


