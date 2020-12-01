"""Perform a fit using a predefined minimizer."""

from gna.ui import basecmd, set_typed
import pickle
from pprint import pprint

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', help='Minimizer to use', metavar='name')
        parser.add_argument('-v', '--verbose', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-s', '--set',     action='store_true', help='Set best fit parameters')
        parser.add_argument('-p', '--push',    action='store_true', help='Set (push) best fit parameters')
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

    __tldr__ =  """\
                The module initializes a fit process with a minimizer, provided by `minimizer-v1`, `minimizer-scan` or others.
                The fit result is saved to the `env.future['fitresult']` as a dictionary.

                Perform a fit using a minimizer 'min':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak_MC --nbins 50 \\
                    -- gaussianpeak --name peak_f  --nbins 50 \\
                    -- ns --name peak_MC --print \\
                          --set E0             values=2    fixed \\
                          --set Width          values=0.5  fixed \\
                          --set Mu             values=2000 fixed \\
                          --set BackgroundRate values=1000 fixed \\
                    -- ns --name peak_f --print \\
                          --set E0             values=2.5  relsigma=0.2 \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 relsigma=0.25 \\
                    -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum \\
                    -- analysis-v1 analysis --datasets peak \\
                    -- stats stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v1 min stats minpars -vv \\
                    -- fit-v1 min \\
                    -- env-print fitresult.min
                ```

                By default the parameters are set to initial after the minimization is done.
                It is possible to set the best fit parameters with option `-s` or with option `-p`.
                The latter option pushed the current values to the stack so they can be recovered in the future.

                The result of the fit may be saved with `save-pickle` or `save-yaml`.

                See also: `minimizer-v1`, `minimizer-scan`.
                """
