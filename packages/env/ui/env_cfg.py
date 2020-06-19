# encoding: utf-8

"""Global environment configuration"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import yaml

class DictWrapperVerbose(DictWrapper):
    def __setitem__(self, k, v):
        if k in self:
            action='Overwrite'
        else:
            action='Set'
        key = '.'.join(k)
        print('{action} {key}: {value}'.format(action=action, key=key, value=v))
        DictWrapper.__setitem__(self, k, v)

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-v', '--verbose', action='count', help='make environment verbose on set')

    def init(self):
        if self.opts.verbose:
            self.env.future = DictWrapperVerbose(self.env.future)
