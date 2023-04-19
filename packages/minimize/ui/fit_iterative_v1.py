"""Perform an iterative fit using a predefined minimizer."""

from typing import Callable
import pickle
from pprint import pprint

import numpy as np

from gna.ui import basecmd
from minimize.lib.base import FitResult

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', help='Minimizer to use', metavar='name')
        parser.add_argument('-v', '--verbose', action='store_true', help='Print fit result to stdout')
        parser.add_argument('-s', '--set', dest='set', action='store_true', help='Set (push) best fit parameters')
        parser.add_argument('-p', '--push', dest='set',    action='store_true', help='Set (push) best fit parameters')
        parser.add_argument('-ch', '--cov-hook', required=True, help='A hook to update covariance matrix after fit')
        parser.add_argument('-cp','--control-pars', nargs='*',
                            help='Fit parameters used to decide when to stop iterative fitting')
        parser.add_argument('--acceptance-treshold', default=0.1, type=float, help='''Sets treshold such that fit
            is accepted when relative difference between to fit results with respect to uncertainties
            gets less then it''')
        parser.add_argument('--profile-errors', '-e', nargs='+', default=[], help='Calculate errors based on statistics profile')
        parser.add_argument('--ndf', type=lambda x: env.future["ndf"][x],
                            help='Read NDF for given chi2 from env')

    def init(self):
        minimizer = self.env.future['minimizer'][self.opts.minimizer]
        self.control_pars = self.opts.control_pars
        self.acceptance_treshold = self.opts.acceptance_treshold
        cov_updater = self.env.future[f'hooks.{self.opts.cov_hook}']

        current_fit, previous_fit = minimizer.fit(), None
        counter = 1
        while self.fit_not_accepted(current_fit, previous_fit):
            self.update_pars(minimizer, current_fit)
            cov_updater()

            previous_fit = current_fit
            current_fit = minimizer.fit()
            counter += 1
        else:
            self.update_pars(minimizer, current_fit)
            cov_updater()


        result = self.result = current_fit

        if self.opts.ndf:
            result['ndf'] = self.opts.ndf

        if self.opts.profile_errors:
            print(f"Running MINOS procedure for re-estimation of uncertainties for {self.opts.profile_errors}")
            minimizer.profile_errors(self.opts.profile_errors, result)

        if not self.opts.set:
            self.reset_pars(minimizer, counter)
            cov_updater()

        if self.opts.verbose:
            print(f"Fit converged after {counter} minimization")
            self.print()

        self.env.future.child('fitresult')[self.opts.minimizer] = result
        minimizer.saveresult(self.env.future.child('fitresults'))

    def print(self):
        print('Fit result for {}:'.format(self.opts.minimizer))
        pprint(dict(self.result))

    def update_pars(self, minimizer, result):
        if result['success']:
            for parspec, value in zip(minimizer.parspecs.specs(), result['x']):
                parspec.par.push(value)
        else:
            raise ValueError('Fit failed: not able to set parameters')

    def reset_pars(self, minimizer, result, stack_size):
        for _ in range(stack_size):
            for parspec in minimizer.parspecs.specs():
                parspec.par.pop()

    def fit_not_accepted(self, current_fit: FitResult, previous_fit: FitResult) -> bool:
        def vals_and_unc(fit):
            keys = self.control_pars if self.control_pars else fit['xdict'].keys()
            vals = np.asarray([fit['xdict'][x] for x in keys])
            unc =  np.asarray([fit['errorsdict'][x] for x in keys])
            return vals, unc

        if previous_fit is None:
            return True

        if any(not x['success'] for x in (previous_fit, current_fit)):
            raise ValueError(f'Fits failed to converged!')

        current_vals, current_unc = vals_and_unc(current_fit)
        previous_vals, _  = vals_and_unc(current_fit)
        distance = np.sqrt(np.square((current_vals - previous_vals)/current_unc).sum())

        return distance > self.acceptance_treshold

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
                    -- analysis-v1 analysis --datasets peak --covariance-updater ana_hook \
                    -- stats stats --chi2 analysis \\
                    -- pargroup minpars peak_f -vv \\
                    -- minimizer-v1 min stats minpars -vv \\
                    -- fit-iterative-v1 min --cov-hook ana --control-pars peak_f.E0 peak_f.Width --set -vv \\
                       --profile-errors peak_f.E0 peak_f.Width \\
                    -- env-print fitresult.min

                ```

                By default the parameters are set to initial after the minimization is done.
                It is possible to set the best fit parameters with option `-s` or with option `-p`.
                The both options push the current values to the stack so they can be recovered in
                the future, it is difference with fit-v1.

                The result of the fit may be saved with `save-pickle` or `save-yaml`.

                See also: `minimizer-v1`, `minimizer-scan`.
                """
