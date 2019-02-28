"""
Simple fit module. Makes the minimizer fit the model.
"""

from __future__ import print_function
from gna.ui import basecmd, set_typed
from gna.configurator import NestedDict

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer), help='Minimizer to use', metavar='name')
        parser.add_argument('-p', '--print', action='store_true', help='Print fit result to stdout')
        # parser.add_argument('-o', '--output', help='Output file (hdf5)', metavar='filename')

    def init(self):
        result = self.result = self.opts.minimizer.fit()

        if self.opts.print:
            self.print()

        # if self.opts.output:
            # self.save()

    def print(self):
        print('Fit result:', end='')
        print(NestedDict(self.result.__dict__))

    # def save(self):
        # try:
            # from h5py import File
        # except Exception as e:
            # raise Exception('Unable to import h5py, required to save the fit result. Try installing python-h5py module')

        # filename = self.opts.output
        # mode = 'w'

        # with File(filename, mode) as ofile:
            # data = self.result.__dict__
            # import IPython; IPython.embed()


