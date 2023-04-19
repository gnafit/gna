"""Saves a subtree of the env to a binary pickle file."""

from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
import pickle
from env.lib.cwd import update_namespace_cwd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='be more verbose')
        parser.add_argument('-o', '--output', required=True, help='Output fil ename')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
        storage = self.env.future
        data = DictWrapper(dict(), split='.')
        for path in self.opts.paths:
            try:
                data[path] = storage[path].unwrap()
            except KeyError:
                raise Exception('Unable to read data path: '+path)

        with open(self.opts.output, 'wb') as ofile:
            pickle.dump(data.unwrap(), ofile, pickle.HIGHEST_PROTOCOL)

        if self.opts.verbose:
            print('Save output file:', self.opts.output)

    __tldr__ = """\
            The module saves the paths provided as arguments to an output pickle file, provided after `-o` option.
            If the outputs should be saved, the data should be converted via `env-data` module.
            The pickle is a binary readable and works fast. It should be preferred over `save-yaml` for the large data.

            The module is similar to the modules `save-yaml` and `save-root`.

            Write the data, collected in the 'output' to the file 'output.pkl':
            ```sh
            ./gna \\
                -- gaussianpeak --name peak --nbins 50 \\
                -- env-data -c spectra output '{note: extra information}' -vv \\
                -- env-print -l 40 \\
                -- save-pickle output -o output.pkl
            ```

            See also: `env-data`, `save-yaml`, `save-root`.
        """
