"""Saves a subtree of the env to a yaml file."""

from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import yaml
import numpy as np
from collections import OrderedDict
from packages.env.lib.cwd import update_namespace_cwd

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
        parser.add_argument('-o', '--output', required=True, help='Output fil ename')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
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

    __tldr__ = """\
            The module saves the paths provided as arguments to an output YAML file, provided after '-o' option.
            If the outputs should be saved, the data should be converted via `env-data` module.

            \033[32mWrite the data from all the outputs from the 'spectra' to 'output':
            \033[31m./gna \\
                -- gaussianpeak --name peak --nbins 50 \\
                -- env-data-root -c spectra.peak output -vv \\
                -- env-print -l 40\033[0m
        """
