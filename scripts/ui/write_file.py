import numpy as np
from gna.dataset import Dataset

pred = self.env.get(self.opts.args[0])
poisson_data = np.random.poisson(pred.data())
print 'Fluctuated data', poisson_data
print 'Write data to', self.opts.args[1]
np.savez(self.opts.args[1], data=poisson_data)

#  print self.opts.args
#  with np.load('/tmp/ex.npz') as ex_file:
    #  a =  ex_file['arr_0']
#  print a

#  dat = Dataset()
#  theory = self.env.get(self.opts.args[0])
#  dat.assign(theory, poisson_data, poisson_data)
#  self.env.dataset
