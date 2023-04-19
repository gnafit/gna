"""Perform a scan using a predefined minimizer."""

from itertools import chain

import numpy as np

from gna.ui import basecmd
from gna.env import env
from env.lib.cwd import update_namespace_cwd
from minimize.lib.pointtree_v01 import PointTree_v01

def _set_attributes(tree, attrs):
    '''Set a list of attributes to each level of HDF5 hierarchy, each
    attribute is set same for each object on the same level.
    attrs[0] is set on first level, attrs[1] is set on second level and it
    continutes recursively.
    '''
    for key in tree.keys():
        node = tree[key]
        node.attrs['parname'] = attrs[0]
        if len(attrs) != 1:
            _set_attributes(node, attrs[1:])

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', help='Minimizer to use', metavar='name')
        parser.add_argument('--output', required=True, help='Path to output HDF5 file')

    def init(self):
        def on_grid(mapping):
            return tuple(mapping[par] for par in scan_pars)

        def from_profiling(mapping):
            return tuple(val for key, val in mapping.items() if key not in scan_pars)

        def make_leaf(leaf):
            return '/'.join(chain((str(num) for num in scanned_values), (leaf,)))

        update_namespace_cwd(self.opts, 'output')
        minimizer = self.env.future['minimizer'][self.opts.minimizer]

        results = minimizer.grid_scan()

        output_store = PointTree_v01(env, self.opts.output, mode='w')

        scan_pars = minimizer.fixed_order
        with output_store:
            output_store.params = scan_pars
            profile_dt = np.dtype([('value', np.float64), ('uncertainty', np.float64)])

            for res in results:
                statistic_in_minimum = res['fun']
                converged = res['success']
                scanned_values = on_grid(res['xdict'])
                profiled_values = from_profiling(res['xdict'])
                profiled_uncertainties = from_profiling(res['errorsdict'])
                profiled = np.fromiter(zip(profiled_values, profiled_uncertainties),
                                       dtype=profile_dt)
                datafit = make_leaf('datafit')
                profiled_params = make_leaf('profiled_params')
                output_store[datafit] = np.array([statistic_in_minimum if converged else np.nan])
                output_store[profiled_params] = profiled
            _set_attributes(output_store, scan_pars)



    __tldr__ =  """\
                The module initializes a scan process with a minimizer, provided by  `minimizer-scan`.
                The scan result is saved to the HDF5 file provided by
                `--output` flag.

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
                    -- pargroup pars peak_f -vv \\
                    -- pargrid scangrid --linspace peak_f.E0 0 10 100  \
                                        --linspace peak_f.Width 0.1 2 100 \
                    -- minimizer-scan min stats pars scangrid -vv \\
                    -- scan-v1 min --output test.hdf5
                ```
                """
