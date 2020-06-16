# -*- coding: utf-8 -*-
"""Save given path within env to the yaml file"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import yaml
import numpy as np
from collections import OrderedDict

def ndarrayRepresenter(dumper, data):
    return dumper.represent_str(repr(data))

def ordereddictRepresenter(dumper, data):
    return dumper.represent_dict(data)

yaml.add_representer(np.ndarray, ndarrayRepresenter)
yaml.add_representer(OrderedDict, ordereddictRepresenter)

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')
        parser.add_argument('-o', '--output', help='Output fil ename')

    def init(self):
        storage = self.env.future
        data = DictWrapper({}, split='.')
        for path in self.opts.paths:
            try:
                data[path] = storage[path].unwrap()
            except KeyError:
                raise Exception('Unable to read data path: '+path)

        with open(self.opts.output, 'w') as ofile:
            ofile.write(yaml.dump(data.unwrap()))

        if self.opts.verbose:
            print('Save output file:', self.opts.output)
