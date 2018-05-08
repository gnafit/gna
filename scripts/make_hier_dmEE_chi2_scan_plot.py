#!/usr/bin/python
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import h5py

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

chi2map_ih = h5py.File('/tmp/out_ih.hdf5', 'r')
chi2map_nh = h5py.File('/tmp/out_nh.hdf5', 'r')

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
points_nh = np.fromiter(chi2map_nh.iterkeys(), dtype=np.float64) * 1000
chi2_nh = np.fromiter((chi2map_nh[key].get('datafit').value for key in chi2map_nh.iterkeys()), dtype=np.float64)
points_ih = np.fromiter(chi2map_ih.keys(), dtype=np.float64) * 1000
chi2_ih = np.fromiter((chi2map_ih[key].get('datafit').value for key in chi2map_ih.iterkeys()), dtype=np.float64)

idx_min = np.argmin(chi2_nh)

ax.plot(points_nh, chi2_nh, label="NH fit to Asimov", alpha=1)
ax.plot(points_ih, chi2_ih, label="IH fit to Asimov", alpha=1)
ax.axhline(y=chi2_nh[idx_min], linestyle='--', alpha=0.5)
ax.annotate(s='', xy=(points_nh[100],chi2_nh[idx_min]), xycoords='data',
             xytext=(points_nh[100],0),  textcoords='data',
             arrowprops=dict(arrowstyle='<->',facecolor='blue'),)
plt.text(points_nh[104], 6, r'$\Delta \chi^2$', fontsize=22)
plt.xlabel(r'$ |\Delta m^2_{ee}|, \, 10^{-3}$ eV${^2}$', fontsize=22)
plt.ylabel(r'$\chi^2$', fontsize=22)
plt.ylim(0, 20)
plt.legend(loc='best')
plt.title(r'$ \chi^2 $ profiles for Asimov data generated for IH', fontsize=20)
plt.tight_layout()
plt.savefig('chi2-nh-ih.pdf')
