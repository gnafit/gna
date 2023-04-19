from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import Schema, Or, And, Optional, Use, isreadable, isrootfile, isascendingarray, haslength, isfilewithext, isnonnegative
from tools.data_load import read_root
from tools.root_helpers import TFileContext
from mpl_tools.root2numpy import get_bin_edges_axis, get_buffer_hist1
from scipy.interpolate import interp1d
import yaml

class bin2bin_uncertainty_v01(TransformationBundle):
    """Implements nuisance parameters for bin-2-bin uncertainties:
        - same uncertainty
        - loaded from ROOT file
    """
    vcfg: dict
    parameters: list
    parnames: list
    relsigmas: np.ndarray
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
        'bundle': object,
        'name': str,
        Optional('label', default=None): str,
        'edges_target': And(Or(np.ndarray, And([float], Use(np.array))), haslength(min=2), isascendingarray),
        'mode': 'relative',
        'uncertainty': Or(
            {
                'root': And(isrootfile, isreadable),
                'hist': str
                },
            {
                'yaml': And(isfilewithext('yaml'), isreadable),
                'uncertainty': str,
                'binwidth': str,
                },
            {
                'uncertainty': And(float, isnonnegative),
                'binwidth': And(float, isnonnegative)
                },
            ),
        Optional('parlabelfmt', default='{name} scale {i: 3d} ({left:.3f}, {right:.3f})'): str,
        Optional('fixed', default=False): bool,
        Optional('errname', default=False): str,
        Optional('errlabel', default='bin2bin errors'): str
        })

    @staticmethod
    def _provides(cfg):
        names = cfg['name'],
        errname = cfg.get('errname')
        if errname:
            names+=errname,
        return (), names

    def build(self):
        name = self.vcfg['name']
        label = self.vcfg['label'] or f'{name} nuisance scale'
        self.vararray = C.VarArray(self.parnames, labels=label)
        self.set_output(name, None, self.vararray.vararray.points)

        errname = self.vcfg.get('errname')
        if errname:
            errlabel = self.vcfg['errlabel']
            self.errarray = C.Points(self.relsigmas, labels=errlabel)
            self.set_output(errname, None, self.errarray.points.points)

    def get_sigmas(self, edges: np.ndarray):
        """Get relative uncertainties for new bin edges.
        The uncertainties are scaled according to the bin width.

        Note:
            - Absolute uncertainties are scaled as abs_new = abs_old * sqrt(width_new/width_old)
            - Relative uncertainties are scaled as rel_new = rel_old * sqrt(width_old/width_new)
              (assuming constant shape within bin)
              """
        uncdef = self.vcfg['uncertainty']

        widths_new = edges[1:]-edges[:-1]
        centers = 0.5*(edges[1:]+edges[:-1])
        if 'yaml' in uncdef:
            with open(uncdef['yaml'], 'r') as f:
                data = yaml.load(f, Loader=yaml.Loader)
            sigma0 = data[uncdef['uncertainty']]
            widths_old = data[uncdef['binwidth']]

            return (widths_old/widths_new)**0.5 * sigma0
        elif 'uncertainty' in uncdef:
            sigma0 = uncdef['uncertainty']
            widths_old = uncdef['binwidth']

            return (widths_old/widths_new)**0.5 * sigma0
        elif 'root' in uncdef:
            with TFileContext(uncdef['root']) as f:
                hist = f.Get(uncdef['hist'])
                edges0: np.ndarray = get_bin_edges_axis(hist.GetXaxis())
                sigmas0 = get_buffer_hist1(hist)
                widths0 = edges0[1:]-edges0[:-1]

                sigma_width_0fcn = interp1d(edges0[:-1], (sigmas0, widths0), kind='previous', fill_value='extrapolate')

                sigmas0_corr, widths0_corr = sigma_width_0fcn(centers)

                ret = (widths0_corr/widths_new)**0.5 * sigmas0_corr
                return ret

        assert False, 'Not implemented (should not be here)'

    def define_variables(self):
        varname = self.vcfg['name']
        parlabelfmt = self.vcfg['parlabelfmt']
        nbins = self.vcfg['edges_target'].size-1
        ndigits = int(np.ceil(np.log10(nbins)))
        edges = self.vcfg['edges_target']
        fixed = self.vcfg['fixed']

        self.parameters=[]
        self.parnames=list(f'{varname}.{varname}_{i:0{ndigits}d}' for i in range(nbins))
        relsigmas = self.relsigmas = self.get_sigmas(edges)
        for i, (parname, left, right, sigma) in enumerate(zip(self.parnames, edges[:-1], edges[1:], relsigmas)):
            label = parlabelfmt.format(name=varname, i=i, left=left, right=right)

            par = self.reqparameter(parname, None, central=1.0, sigma=sigma, label=label, fixed=fixed)
            self.parameters.append(par)

        # self.namespace.printparameters(labels=True)
