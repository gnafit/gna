#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

x = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0])
y = np.array([0.3, 1.8, 3.1, 6.0, 7.7, 9.6, 11.4])
y2= y**2
xfine = np.linspace(x[0], x[-1], 200)

f1_c = np.polyfit(x, y, 2)
f1 = np.poly1d(f1_c)

f2_c = np.polyfit(x, y2, 1)
f2 = np.poly1d(f2_c)

fig = plt.figure()
ax = plt.subplot(111, xlabel='', ylabel='', title='')
ax.minorticks_on()
ax.grid()
ax.plot(x,y, 'o')
ax.plot([0.0, 12.0], [0.0, 12.0], '--', color='gray')

ax.plot(xfine, f1(xfine), label='f1')
ax.plot(xfine, f2(xfine)**0.5, label='f2')

ax.legend()

plt.show()

print('F1', f1_c)
print('F2', f2_c)
