"""Save given path within env to the pickle file"""

from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
from collections import OrderedDict
import pickle
from packages.env.lib.cwd import update_namespace_cwd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')
        parser.add_argument('-o', '--output', required=True, help='Output fil ename')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
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
