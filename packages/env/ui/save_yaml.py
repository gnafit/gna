"""Simple fit module. Makes the minimizer fit the model."""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import yaml

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')
        parser.add_argument('-o', '--output', help='Output fil ename')

    def init(self):
        data = DictWrapper({}, split='.')
        for path in self.opts.paths:
            try:
                data[path] = self.env.future[path]
            except KeyError:
                raise Exception('Unable to read data path: '+path)

        with open(self.opts.output, 'w') as ofile:
            ofile.write(yaml.dump(data.unwrap()))

        if self.opts.verbose:
            print('Save output file:', self.opts.output)
