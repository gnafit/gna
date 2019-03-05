import numpy as np
from gna.dataset import Dataset

x = np.load('/tmp/peakdata.npz')['data']
peak_fakedata = Dataset()
peak_fakedata.assign(self.env.get('peak/spectrum'), x, x)
self.env.parts.dataset['peak_fakedata'] = peak_fakedata
