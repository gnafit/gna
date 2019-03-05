import numpy as np
from gna.dataset import Dataset

# Usage 0 where read from, 1 to which observable assign, 2 name for dataset

data = np.load(self.opts.args[0])['data']
print "Read data from {0}, it is {1}".format(self.opts.args[0], data)
theory = self.env.get(self.opts.args[1])
tmp = Dataset()
tmp.assign(theory, data, data)
print "Creating dataset with theory {0} and data".format(self.opts.args[1])
self.env.parts.dataset[self.opts.args[2]] = tmp
print "Storing dataset in env as {}".format(self.opts.args[2])
