"""Save given path within env to the pickle file"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
from collections import OrderedDict
import pickle

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')
        parser.add_argument('-o', '--output', required=True, help='Output fil ename')

    def init(self):
        storage = self.env.future
        data = DictWrapper(OrderedDict(), split='.')
        for path in self.opts.paths:
            try:
                data[path] = storage[path].unwrap()
            except KeyError:
                raise Exception('Unable to read data path: '+path)

        with open(self.opts.output, 'wb') as ofile:
            pickle.dump(data.unwrap(), ofile, pickle.HIGHEST_PROTOCOL)

        if self.opts.verbose:
            print('Save output file:', self.opts.output)
