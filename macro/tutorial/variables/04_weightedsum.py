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
w1 = ns.defparameter('w1', central=1.0, free=True, label='weight 1')
w2 = ns.defparameter('w2', central=0.1, free=True, label='weight 2')
w3 = ns.defparameter('w3', central=0.1, free=True, label='weight 3')

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Create x input
x = np.linspace(-1.0*np.pi, 1.0*np.pi, 500)
a1 = C.Points(np.sin(x))
a2 = C.Points(np.sin(16.0*x))
a3 = C.Points(np.cos(16.0*x))
outputs = [a.points.points for a in (a1, a2, a2)]

weights = ['w1', 'w2', 'w3']
wsum = C.WeightedSum(weights, outputs)
wsum.print()

fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'f(x)' )
ax.set_title( r'$a\sin(x)+b\sin(16x)+c\cos(16x)$' )

label = 'a={}, b={}, c={}'.format(w1.value(), w2.value(), w2.value())
wsum.sum.sum.plot_vs(x, label=label)

w1.push(0.0)
label = 'a={}, b={}, c={}'.format(w1.value(), w2.value(), w2.value())
wsum.sum.sum.plot_vs(x, label=label)

w1.pop()
w2.push(0.0)
label = 'a={}, b={}, c={}'.format(w1.value(), w2.value(), w2.value())
wsum.sum.sum.plot_vs(x, label=label)

w2.pop()
w3.push(0.0)
label = 'a={}, b={}, c={}'.format(w1.value(), w2.value(), w2.value())
wsum.sum.sum.plot_vs(x, label=label)

ax.legend(loc='upper right')

plt.show()
