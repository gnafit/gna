"""Global environment configuration UI. Enables verbosity for the debugging purposes."""

from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import yaml

class DictWrapperVerbose(DictWrapper):
    def __init__(self, dct, exclude=[], include=[], **kwargs):
        parent = kwargs.get('parent')
        DictWrapper.__init__(self, dct, **kwargs)
        if parent:
            self._exclude=parent._exclude
            self._include=parent._include

            path = parent._path
            for k, v in parent._obj.items():
                if v is dct:
                    self._path = parent._path + (k,)
                    break
            else:
                self._path=tuple()
        else:
            self._exclude=exclude
            self._include=include
            self._path=tuple()

    def _skip(self, key):
        if any(excl in key for excl in self._exclude):
            return True

        if not self._include:
            return False

        if any(incl in key for incl in self._include):
            return False

        return True

    def __setitem__(self, k, v):
        if k in self:
            action='Overwrite'
        else:
            action='Set'

        DictWrapper.__setitem__(self, k, v)

        key = self._path + tuple(self.iterkey(k))
        if self._skip(key):
            return

        key = '.'.join(key)
        print('{action} {key}: {value}'.format(action=action, key=key, value=v))

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-v', '--verbose', action='count', default=0, help='make environment verbose on set')
        parser.add_argument('-x', '--exclude', nargs='+', default=[], help='keys to exclude')
        parser.add_argument('-i', '--include', nargs='+', default=[], help='keys to include (only)')

    def init(self):
        if self.opts.verbose:
            self.env.future = DictWrapperVerbose(self.env.future, include=self.opts.include, exclude=self.opts.exclude)

    __tldr__ = {
            "" : """\
                All assignments and changes of the environment will be printed to stdout.

                \033[32mEnable verbosity:
                \033[31m./gna \\
                    -- env-cfg -v \\
                    -- gaussianpeak --name peak_MC --nbins 50\033[0m

                The output may be filtered with `-x` and `-i` keys. Both support multiple arguments.

                \033[32mThe `-x` option excludes matching keys:
                \033[31m./gna \\
                    -- env-cfg -v -x fcn \\
                    -- gaussianpeak --name peak_MC --nbins 50\033[0m

                \033[32mThe `-i` option includes matching keys exclusively:
                \033[31m./gna \\
                    -- env-cfg -v -i spectrum \\
                    -- gaussianpeak --name peak_MC --nbins 50\033[0m

                See also: `env-print`, `env-set`.
            """,
            }
