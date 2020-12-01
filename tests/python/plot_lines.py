#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_tools.plot_lines import plot_lines

fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( '' )

lines = (
        'first line with formula $e=mc^2$',
        'second line with number: 240'
        )
plot_lines('text', loc='upper right')

fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( '' )
ax.set_ylabel( '' )
ax.set_title( '' )
plt.subplots_adjust(right=0.6)

lines = (
        'first line with formula $e=mc^2$',
        'second line with number: 240'
        )
plot_lines(lines, loc='upper right', outside=[1.02, 1.0])

plt.show()
