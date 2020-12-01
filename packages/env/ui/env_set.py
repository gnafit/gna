"""Assigns any data within env.
Needed to provide an extra information to be saved with `save-yaml` and `save-pickle`."""

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
        parser.add_argument('-a', '--append', nargs=2, action='append', metavar=('KEY', 'VALUE'), default=[], help='add custom fields to the output')
        parser.add_argument('-y', '--yaml', dest='append_yaml', nargs=2, action='append', metavar=('KEY', 'YAML'), default=[], help='add custom fields to the (value parsed by yaml)')
        parser.add_argument('update_yaml', nargs='*', type=yamlload, metavar=('YAMLDICT'), help='yaml input to update the dictionary')

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

    __tldr__ = """\
                The module provides three ways to input data:
                1. Update env from a dictionary (nested), defined via YAML.
                2. Write a string to an address within env.
                3. Write parsed YAML to an address within env.

                Optional argument `-r` may be used to set root address.

                Write two key-value pairs to the 'test':
                ```sh
                ./gna \\
                    -- env-set -r test '{key1: string, key2: 1.0}' \\
                    -- env-print test
                ```
                The first value, assigned by the key 'key1' is a string 'string', the second value is a float 1.

                The `-y` argument may be used to write a key-value pair:
                ```sh
                ./gna \\
                    -- env-set -r test -y sub '{key1: string, key2: 1.0}' \\
                    -- env-print test
                ```
                The command does the same, but writes the key-value pairs into a nested dictionary under the key 'sub'.

                The `-a` argument simply writes a key-value pair, where value is a string:
                ```sh
                ./gna \\
                    -- env-set -r test -a key1 string \\
                    -- env-print test
                ```

                See also: `env-print`, `env-cfg`.
            """
