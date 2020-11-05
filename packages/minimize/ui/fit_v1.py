"""Simple fit module. Makes the minimizer fit the model."""

from __future__ import print_function
from gna.ui import basecmd, set_typed
import pickle
from pprint import pprint

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', help='Minimizer to use', metavar='name')
        parser.add_argument('-v', '--verbose', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-s', '--set',   action='store_true', help='Set best fit parameters')
        parser.add_argument('-p', '--push',   action='store_true', help='Set (push) best fit parameters')
        parser.add_argument('--profile-errors', '-e', nargs='+', default=[], help='Calculate errors based on statistics profile')
        parser.add_argument('--simulate', action='store_true', help='do nothing')

    def init(self):
        minimizer = self.env.future['minimizer'][self.opts.minimizer]
        if self.opts.simulate:
            return

        result = self.result = minimizer.fit(profile_errors=self.opts.profile_errors)

        if self.opts.set or self.opts.push:
            push = self.opts.push
            if result['success']:
                for parspec, value in zip(minimizer.parspecs.specs(), result['x']):
                    if push:
                        parspec.par.push(value)
                    else:
                        parspec.par.set(value)
            else:
                print('Fit failed: not setting parameters')

        if self.opts.verbose:
            self.print()

        self.env.future.child('fitresult')[self.opts.minimizer] = result
        minimizer.saveresult(self.env.future.child('fitresults'))

    def print(self):
        print('Fit result for {}:'.format(self.opts.minimizer))
        pprint(dict(self.result))
