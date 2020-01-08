# encoding: utf-8
"""Fit implementation, that scans Δm² and fits all the other parameters on each iteration.

Does the scan, shows the figure. Nothing more yet."""

from __future__ import print_function
from gna.ui import basecmd, set_typed
from gna.configurator import NestedDict
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer), help='Minimizer to use', metavar='name')
        parser.add_argument('parameter', help='Δm² parameter name')
        parser.add_argument('-l', '--limits', nargs=3, required=True, type=float, help='Δm² limits and step', metavar=('min', 'max', 'step'))
        parser.add_argument('-p', '--print', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-s', '--set',   action='store_true', help='Set best fit parameters')
        parser.add_argument('--profile-errors', '-e', nargs='+', default=[], help='Calculate errors based on statistics profile')
        parser.add_argument('-o', '--output', help='Output file (yaml)', metavar='filename')
        parser.add_argument('-a', '--append', nargs=2, action='append', default=[], help='add custom fields to the output')
        parser.add_argument('-v', '--verbose', action='count', help='verbosity')

    def init(self):
        self.minimizer = self.opts.minimizer
        steps = np.arange(*self.opts.limits)
        statuses = []
        self.chi2 = np.ma.array(np.zeros_like(steps), mask=np.zeros_like(steps, dtype='bool'))
        for i, value in enumerate(steps):
            if self.opts.verbose>1:
                print('Scanning {} {:3d}: {}'.format(self.opts.parameter, i, value))

            self.minimizer.fixpar(self.opts.parameter, value)
            result = self.minimizer.fit()
            statuses.append(result)
            self.chi2[i] = result.fun
            self.chi2.mask[i] = not result.success

        from matplotlib import pyplot as plt
        plt.plot(steps, self.chi2)
        plt.show()

    def final(self):
        if self.opts.set and result.success:
            for par, value in zip(minimizer.pars, result.x):
                par.set(value)

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
            data = self.result.__dict__.copy()
            for key in ('errorsdict', 'errors_profile', 'xdict'):
                if key in data:
                    data[key] = dict(data[key])

            if self.opts.append:
                data.update(self.opts.append)

            ofile.write(yaml.dump(data))
