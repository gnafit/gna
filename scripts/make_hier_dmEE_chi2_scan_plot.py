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

min_chi2_nh, min_chi2_ih = np.min(chi2_nh), np.min(chi2_ih)
if min_chi2_nh > min_chi2_ih:
    idx_min = np.argmin(min_chi2_ih)
    upper_pos = min_chi2_nh
else:
    idx_min = np.argmin(min_chi2_nh)
    upper_pos = min_chi2_ih


ax.plot(points_nh, chi2_nh, label="NH fit to Asimov", alpha=1)
ax.plot(points_ih, chi2_ih, label="IH fit to Asimov", alpha=1)

#  import IPython
#  IPython.embed()

line = ax.axhline(y=upper_pos, linestyle='--', alpha=0.5)
ax.annotate(s='', xy=(points_nh[100], upper_pos), xycoords='data',
             xytext=(points_nh[100],0),  textcoords='data',
             arrowprops=dict(arrowstyle='<->', color=line.get_color(), alpha=0.5),)
plt.grid(alpha=0.4)
plt.text(points_nh[104], 6, r'$\Delta \chi^2$', fontsize=22)
plt.xlabel(r'$ |\Delta m^2_{ee}|, \, 10^{-3}$ eV${^2}$', fontsize=20)
plt.ylabel(r'$\chi^2$', fontsize=20)
plt.ylim(0, 20)
plt.legend(loc='best')
plt.title(r'$ \chi^2 $ profiles for Asimov data generated for NH', fontsize=20)
plt.tight_layout()
plt.savefig('chi2-nh-ih.pdf')
