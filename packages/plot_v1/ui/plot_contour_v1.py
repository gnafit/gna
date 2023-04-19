import argparse
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np

from scipy.optimize import bisect

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
import matplotlib.patches as mpatches

from gna.ui import basecmd
from gna.env import env
from mpl_tools.helpers import savefig
from plot_v1.lib.contour_utils import pvmaptypes, Chi2Reader
from plot_v1.lib.fit_extractor import FitExtractor
from plot_v1.lib.cl_specs import parselevel, criticalvalue
from plot_v1.lib.contour_saver import ContourSaver


class MplStyleSheet(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        builtin_styles = plt.style.available
        if values in builtin_styles:
            setattr(namespace, self.dest, values)
        else:
            rc_file = Path(values)
            if rc_file.is_file():
                setattr(namespace, self.dest, rc_file)
            else:
                raise argparse.ArgumentTypeError(f'''{values} is neither a matplotlib builtin style nor an external stylesheet file''')


class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-ts', '--test-statistic-map', type=Path,
                            help='Path to HDF5 file with test statistic map')
        parser.add_argument('--plot', dest='plots', nargs='+',
                            action='append', required=True)
        parser.add_argument('--bestfit', type=FitExtractor, help='Best fit results from minimizer')
        parser.add_argument('--no-bf', action='store_true', help='Do not plot best fit point')
        parser.add_argument('--labels', action='append', help='Optional list of labels')
        parser.add_argument('-rc', '--mplrc', action=MplStyleSheet, help='A matplotlib builtin or externaly provided style')
        parser.add_argument('--legend', action='store_true', help='Draw legend. Mandatory for drawing filled contours')
        parser.add_argument('-ct', '--contour-type', default='line', choices=['line', 'filled'],
                            help='Choose contour type, plain lines or filled')
        parser.add_argument('--save-contour', type=Path, help="Path where contours will be saved. Can be root or HDF currently")

    def extract_bestfit_info(self) -> float:
        '''Extract test statistic value in best fit point, parameter values
        and uncertainties'''
        bestfit_results = self.opts.bestfit.fit_results
        test_statistic_min = bestfit_results['fun']
        assert bestfit_results['success'], "Best fit was not found!"
        minos_errors = bestfit_results.get('errors_profile')
        minos_errors = minos_errors.unwrap() if minos_errors else None
        for par, best_value in bestfit_results['xdict'].items():
            self.bestfits[par]['bestfit'] = best_value
        for par, errors in bestfit_results['errorsdict'].items():
            if minos_errors and par in minos_errors:
                self.bestfits[par]['uncertainty'] = np.array([[abs(x) for x in minos_errors[par]]]).T
            else:
                self.bestfits[par]['uncertainty'] = errors
        return test_statistic_min

    def run(self):
        self.params = None
        self.bestfits = defaultdict(dict)
        self.ndim = None
        self.labels = iter(self.opts.labels) if self.opts.labels else None

        test_statistic_in_best_fit = self.extract_bestfit_info()
        test_statistic_map = Chi2Reader(self.opts.test_statistic_map)
        self.params = test_statistic_map.params
        if self.opts.mplrc:
            plt.style.use(self.opts.mplrc)

        ax = plt.subplot()

        self.colors = list('bgrcmyk')
        for plotdesc in self.opts.plots:
            plottype, rest = plotdesc[0], plotdesc[1:]
            pvmap = pvmaptypes[plottype](reader=test_statistic_map,
                                         bestfit=test_statistic_in_best_fit)
            pvspecs = [(pvspec, parselevel(pvspec)) for pvspec in rest]
            crspecs = [(pvspec, criticalvalue(pvspec)) for pvspec in rest]
            if pvmap.ndim == 1:
                contours, bestfit = self.plot_1dlevels(ax, pvmap, crspecs, test_statistic_map)
            elif pvmap.ndim == 2:
                if not self.opts.no_bf:
                    bestfit = self.plot_bestfit(ax)
                contours = self.plot_levels(ax, pvmap, pvspecs)
                if self.opts.save_contour:
                    ContourSaver(output=self.opts.save_contour, contours=contours,
                             bestfit=bestfit)

    def plot_bestfit(self, ax):
        def val_and_unc(x):
            par_info = self.bestfits[x]
            return par_info['bestfit'], par_info['uncertainty']

        x, y = [_.qualifiedName() for _ in reversed(self.params)]
        x_val, x_unc = val_and_unc(x)
        y_val, y_unc = val_and_unc(y)
        ax.errorbar(x=x_val, y=y_val, xerr=x_unc, yerr=y_unc, color='black',
                marker='o', capsize=2)
        return [(x_val, x_unc), (y_val, y_unc)]

    def plot_1dlevels(self, ax, pvmap, crspecs, test_statistic_map):
        def val_and_unc(x):
            par_info = self.bestfits[x]
            return par_info['bestfit'], par_info['uncertainty']

        def plot_profile(left_bound_interp, right_bound_interp):
            if left_bound == left_bound_interp:
                ax.plot([left_bound, right_bound, right_bound],
                        [crlevel, crlevel, 0], color='black', linestyle='--', alpha=0.6)
            elif right_bound == right_bound_interp:
                ax.plot([left_bound, left_bound, right_bound],
                        [0, crlevel, crlevel], color='black', linestyle='--', alpha=0.6)
            else:
                ax.plot([left_bound, left_bound, right_bound, right_bound],
                        [0, crlevel, crlevel, 0], color='black', linestyle='--', alpha=0.6)

        def calc_bound(first_bound, second_bound):
            bound = 0.
            try:
                bound = bisect(func, first_bound, second_bound)
            except ValueError:
                print(f"Couldn't build bound of {crlevel:1.2f} chi2 level")
                bound = first_bound
            if log:
                return 10**bound
            return bound

        par_name = self.params[0].qualifiedName()
        par_val, par_unc = val_and_unc(par_name)
        crlevels = []
        order = lambda x: x[1]
        for spec, level in sorted(crspecs, key=order):
            crlevels.append(level)
        XX = pvmap.data.grids[0]
        YY = pvmap.dchi2
        log = False
        interp1d = test_statistic_map.interp1d
        if XX[0] - XX[1] < 1.e-17:
            log = True
            XX = np.log10(XX)
            interp1d = test_statistic_map.interp1d_log
        left_bound_interp = interp1d.x[0]
        right_bound_interp = interp1d.x[-1]
        local_func_min_x = XX[np.where(YY == YY.min())]
        for i, crlevel in enumerate(crlevels):
            func = lambda x: interp1d(x) - crlevel - pvmap.bestfit
            left_bound = calc_bound(left_bound_interp, local_func_min_x)
            right_bound = calc_bound(right_bound_interp, local_func_min_x)
            if log:
                plot_profile(10**left_bound_interp, 10**right_bound_interp)
            else:
                plot_profile(left_bound_interp, right_bound_interp)
        return None, (par_val, par_unc)


    def plot_levels(self, ax, pvmap, pvspecs):
        def plot_filled():
            CS = ax.contourf(XX, YY, pvmap.data, levels=pvlevels)
            if self.opts.legend or self.opts.labels:
                layers = CS.collections
                colors = [layer.get_facecolor().ravel() for layer in layers]
                labels = self.opts.labels if self.opts.labels else specs
                patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
                ax.legend(handles=patches, loc='best')
                return CS

        def plot_lines():
            CS = ax.contour(XX, YY, pvmap.data, levels=pvlevels)
            for c, spec in zip(CS.collections, specs):
                if self.labels:
                    try:
                        c.set_label(next(self.labels))
                    except StopIteration:
                        c.set_label(f"{pvmap.name} {spec}")
                else:
                    c.set_label(f"{pvmap.name} {spec}")
            return CS


        ndim = pvmap.ndim
        pvlevels = []
        specs = []
        order = lambda x: x[1]
        for spec, level in sorted(pvspecs, key=order):
            pvlevels.append(level)
            specs.append(spec)
        else:
            pvlevels.append(1)

        XX, YY = np.meshgrid(*pvmap.data.grids)
        if self.opts.contour_type == 'line':
            return plot_lines()
        else:
            return plot_filled()

    __tldr__='''python3 -- ./gna \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- gaussianpeak --name peak_f  --nbins 50 \
    -- ns --name peak_MC --print \
          --set E0             values=2    fixed \
          --set Width          values=0.5  fixed \
          --set Mu             values=2000 fixed \
          --set BackgroundRate values=1000 fixed \
    -- ns --name peak_f --print \
          --set E0             values=2.5  relsigma=0.2 \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 relsigma=0.25 \
    -- pargroup free_pars peak_f -vv \
    -- pargrid scangrid --linspace peak_f.E0 1.5 2.5 100 \
                        --linspace peak_f.Width 0.4 0.6 100 \
    -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum \
    -- analysis-v1 analysis --datasets peak \
    -- stats stats --chi2 analysis \
    -- minimizer-v1 bestfit_min stats free_pars -vv \
    -- minimizer-scan scan_min stats free_pars scangrid -vv \
    -- fit-v1 bestfit_min --set -v \
    -- scan-v1 scan_min --output testing.hdf5 \
    -- plot-contour-v1 -ts testing.hdf5 --plot chi2ci 1s 2s 3s --bestfit bestfit_min \
    -- mpl-v1 --xlabel E0 --ylabel Width --legend --show
    '''
