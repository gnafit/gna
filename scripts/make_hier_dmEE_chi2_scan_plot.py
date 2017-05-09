#!/bin/python
import matplotlib.pyplot as plt
import h5py
chi2map_ih = h5py.File('/tmp/out_ih.hdf5', 'r')
chi2map_nh = h5py.File('/tmp/out_nh.hdf5', 'r')

points_nh = np.fromiter(chi2map_nh.iterkeys(), dtype=np.float64)
chi2_nh = np.fromiter((chi2map_nh[key].get('datafit').value for key in self.chi2map_nh.iterkeys()), dtype=np.float64)
points_ih = np.fromiter(chi2map_ih.keys(), dtype=np.float64)
chi2_ih = np.fromiter((chi2map_ih[key].get('datafit').value for key in chi2map_ih.iterkeys()), dtype=np.float64)
plt.plot(points_nh, chi2_nh, label="NH fit IH")
plt.plot(points_ih, chi2_ih, label="IH fit IH")
plt.ylim(0, 20)
plt.savefig('chi2-nh-ih.pf')
