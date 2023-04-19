"""Load values of parameters"""

from gna.ui import basecmd
from pylib.tools.cfg_load import cfg_load

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('input', help='file to read the name/value pairs (yaml/pkl)')
        parser.add_argument('--root', default='', help='root of the parameters to set')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
        parser.add_argument('--replace-1', help='string to replace the first element of path of each variable')

    def init(self):
        loadverbose = self.opts.verbose
        if loadverbose>1: loadverbose-=1
        data = cfg_load(self.opts.input, self.opts.verbose)

        data_new = {}
        if self.opts.replace_1:
            for name, value in data.items():
                els = name.split('.')
                els[0]=self.opts.replace_1
                name = '.'.join(els)
                data_new[name] = value
            data = data_new

        env = self.env.globalns(self.opts.root)
        for name, value in data.items():
            if self.opts.verbose>1:
                print(f'  {name}: {value}')
            var=env[name]
            var.set(value)
