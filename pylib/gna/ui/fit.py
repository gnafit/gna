"""Simple fit module. Makes the minimizer fit the model."""

from __future__ import print_function
from gna.ui import basecmd, set_typed
from gna.configurator import NestedDict

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer), help='Minimizer to use', metavar='name')
        parser.add_argument('-p', '--print', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-o', '--output', help='Output file (yaml)', metavar='filename')
        # parser.add_argument('-o', '--output', help='Output file (yaml/hdf5)', metavar='filename')

    def init(self):
        result = self.result = self.opts.minimizer.fit()

        if self.opts.print:
            self.print()

        ofile = self.opts.output
        if ofile:
            if ofile.endswith('.yaml'):
                self.save_yaml(ofile)
            # elif ofile.endswith('.hdf5'):
                # self.save_hdf5(ofile)
            else:
                raise Exception('Unsupported output format or '+ofile)
            print('Save output file:', ofile)

    def print(self):
        print('Fit result:', end='')
        print(NestedDict(self.result.__dict__))

    # def save_hdf5(self, filename):
        # from h5py import File

        # mode = 'w'

        # with File(filename, mode) as ofile:
            # data = self.result.__dict__
            # import IPython; IPython.embed()

    def save_yaml(self, filename):
        import yaml
        mode='w'

        with open(filename, mode) as ofile:
            data = self.result.__dict__
            ofile.write(yaml.dump(data))
