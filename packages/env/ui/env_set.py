"""Print given path within env"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
import yaml

def yamlload(s):
    ret = yaml.load(s, Loader=yaml.Loader)
    return ret

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-r', '--root', help='root environment')
        parser.add_argument('-a', '--append', nargs=2, action='append', default=[], help='add custom fields to the output')
        parser.add_argument('-y', '--yaml', dest='append_yaml', nargs=2, action='append', default=[], help='add custom fields to the (value parsed by yaml)')
        parser.add_argument('update_yaml', nargs='*', type=yamlload, help='yaml input to update the dictionary')

    def init(self):
        storage = self.env.future
        if self.opts.root:
            storage = storage.child(self.opts.root)

        for k, v in self.opts.append:
            storage[k] = v

        for k, v in self.opts.append_yaml:
            storage[k] = yamlload(v)

        for yaml in self.opts.update_yaml:
            storage.update(yaml)
