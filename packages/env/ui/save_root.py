"""Saves a subtree of the env to a binary ROOT file."""

from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
from load import ROOT as R
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
        verbose = self.opts.verbose

        if verbose>1:
            print('Create output file:', self.opts.output)
        output = R.TFile(self.opts.output, 'RECREATE')
        for path in self.opts.paths:
            try:
                st = storage[path]
            except KeyError:
                raise Exception('Unable to read data path: '+path)

            saveobjects(output, st, verbose)

        if verbose:
            print('Save output file:', self.opts.output)
            if verbose>1:
                output.ls()
        output.Close()

def saveobjects(odir, obj, verbose):
    if isinstance(obj, R.TObject):
        if verbose>1:
            print('Write', str(obj))
        odir.WriteTObject(obj, obj.GetName(), 'overwrite')
        return

    if isinstance(obj, DictWrapper):
        for k, v in obj.walkitems():
            path, name = k[:-1], k[-1]
            if path:
                path = '/'.join(path)
                subdir = odir.GetDirectory(path) or odir.mkdir(path)
            else:
                subdir = odir
            saveobjects(subdir, v, verbose)
        return

    print('Unable to save the object to ROOT file, skip:', type(obj))

cmd.__tldr__ = """\
               The module saves the paths provided as arguments to an output ROOT file, provided after `-o` option.
               The outputs that should be saved should be converted via `env-data-root` module.

               The module is similar to the modules `save-yaml` and `save-pickle`.

               Write the data, collected in the 'output' to the file 'output.root':
               ```sh
               ./gna \\
                   -- gaussianpeak --name peak --nbins 50 \\
                   -- env-data-root -c spectra output \\
                   -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \\
                   -- env-print -l 40 \\
                   -- save-root output -o output.root
               ```

               See also: `env-data-root`, `save-yaml`, `save-pickle`.
               """
