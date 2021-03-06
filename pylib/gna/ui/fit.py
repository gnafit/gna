"""Simple fit module. Makes the minimizer fit the model."""

from gna.ui import basecmd, set_typed
from gna.configurator import NestedDict
import pickle

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer), help='Minimizer to use', metavar='name')
        parser.add_argument('-p', '--print', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-s', '--set',   action='store_true', help='Set best fit parameters')
        parser.add_argument('--profile-errors', '-e', nargs='+', default=[], help='Calculate errors based on statistics profile')
        parser.add_argument('-o', '--output', nargs='+', help='Output file(s) (yaml, pickle)', metavar='filename')
        parser.add_argument('-a', '--append', nargs=2, action='append', default=[], help='add custom fields to the output')
        parser.add_argument('--simulate', action='store_true', help='do nothing')

    def init(self):
        minimizer = self.opts.minimizer
        if self.opts.simulate:
            return

        result = self.result = minimizer.fit(profile_errors=self.opts.profile_errors)

        if self.opts.set and result.success:
            for par, value in zip(minimizer.pars, result.x):
                par.set(value)

        if self.opts.print:
            self.print()

        ofile = self.opts.output
        if ofile:
            self.save(ofile)

    def print(self):
        print('Fit result:', end='')
        print(NestedDict(self.result.__dict__))

    def save(self, filenames):
        import yaml
        mode='w'

        data = self.result.__dict__.copy()
        if self.opts.append:
            data.update(self.opts.append)

        for filename in filenames:
            if filename.endswith('.yaml'):
                odata=data.copy()
                for key in ('errorsdict', 'errors_profile', 'xdict'):
                    if key in odata:
                        odata[key] = dict(odata[key])
                with open(filename, mode) as ofile:
                    ofile.write(yaml.dump(odata))
            elif filename.endswith('.pkl'):
                with open(filename, mode+'b') as ofile:
                    pickle.dump(data, ofile, pickle.HIGHEST_PROTOCOL)
            else:
                raise Exception('Unsupported output format or '+filename)

            print('Save output file:', filename)
