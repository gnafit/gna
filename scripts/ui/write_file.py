import numpy as np
from gna.dataset import Dataset

# 0 - which observable to take, 1 - file to write to, 2 - do poisson fluctuations
# or no

pred = self.env.get(self.opts.args[0])
data = None 
if len(self.opts.args) > 2:
    data = np.random.poisson(pred.data())
    print 'Fluctuated data', data
else:
    data = pred.data()
    print 'Data with no fluctuation {}'.format(data)

print 'Write data to', self.opts.args[1]
np.savez(self.opts.args[1], data=data)

#  print self.opts.args
#  with np.load('/tmp/ex.npz') as ex_file:
    #  a =  ex_file['arr_0']
#  print a

#  dat = Dataset()
#  theory = self.env.get(self.opts.args[0])
#  dat.assign(theory, poisson_data, poisson_data)
#  self.env.dataset
