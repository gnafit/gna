#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env
import constructors as C
import numpy as np
from matplotlib import pyplot as plt
from gna.bindings import common

# Make a variable for global namespace
ns = env.globalns

# Create a parameter in the global namespace
ns.defparameter('a', central=1.0,  free=True, label='weight 1 (global)')
ns.defparameter('b', central=-0.1, free=True, label='weight 2 (global)')
ns.defparameter('group.a', central=0.5,  free=True, label='weight 1 (local)')
ns.defparameter('group.b', central=0.00, free=True, label='weight 2 (local)')
ns.defparameter('group.c', central=0.05, free=True, label='weight 3 (local)')

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Create x  and several functions to be added with weights
x = np.linspace(-1.0*np.pi, 1.0*np.pi, 500)
a1 = C.Points(np.sin(x))
a2 = C.Points(np.sin(16.0*x))
a3 = C.Points(np.cos(16.0*x))
outputs = [a.points.points for a in (a1, a2, a3)]

# Initialize the WeightedSum with list of variables and list of outputs
weights1 = ['a', 'b', 'group.c']
wsum1 = C.WeightedSum(weights1, outputs)

weights2 = ['a', 'b', 'c']
with ns('group'):
    wsum2 = C.WeightedSum(weights2, outputs)

wsum1.print(); print()
wsum2.print(); print()

# Do some plotting
fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title( r'$a\,\sin(x)+b\,\sin(16x)+c\,\cos(16x)$' )

wsum1.sum.sum.plot_vs(x, label='Weighted sum 1: a={a}, b={b}, c={c}'.format(**wsum1.variablevalues()))
wsum2.sum.sum.plot_vs(x, label='Weighted sum 2: a={a}, b={b}, c={c}'.format(**wsum2.variablevalues()))

ax.legend(loc='lower right')

from mpl_tools.helpers import savefig
from sys import argv
oname = argv[0].rsplit('/', 1).pop().replace('.py', '.png')
savefig('output/tutorial/'+oname)

plt.show()
