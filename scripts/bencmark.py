#!/usr/bin/python

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

odir = 'output/17.06.19_eres_sparse/'
try:
    os.makedirs(odir)
except:
    pass

par_range = np.linspace(0.01, 0.03, 500)
# Collect execution time in files
for par in par_range:
    old = self.env.get(self.opts.args[0]+'/AD1').data()
    new_sparse = self.env.get(self.opts.args[0]+'/AD1_sparse').data()
    self.env.parameters[self.opts.args[0]+'.detectors.AD1.Eres_b'].set(par)

bench_old = pd.read_csv('bench_old.txt', header=None, names=["Old"])
bench_sparse= pd.read_csv('bench_sparse.txt', header=None, names=["Sparse"])

joint = pd.concat([bench_old, bench_sparse], ignore_index=True, axis=1, names=['Old', 'Sparse'])
joint.rename(columns={0:'Old', 1:'Sparse'}, inplace=True)

fig, ax = plt.subplots()
plt.xlim(0, 30)
joint.plot.hist(ax=ax, alpha=0.5, stacked=False, bins=5000)
plt.legend()
plt.xlabel('Time in microseconds')
plt.title('Time of applying resolutions, 500 samples')
plt.savefig(odir+'bench_resolution.pdf')

print('Mean time:')
print( joint.mean() )

print()
print('Median time:')
print( joint.median() )

