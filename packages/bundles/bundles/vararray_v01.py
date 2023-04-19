from load import ROOT as R
import gna.constructors as C
from gna.configurator import StripNestedDict
from tools.schema import *
from gna.bundle import TransformationBundle
import itertools as it
import numpy as np

class vararray_v01(TransformationBundle):
    """Implements a VarArray v01:
        - a set of parameters
            + usable for the minimizer
            + unusable for the expression (not announced)
        - relevant VarArray output
        - indices are not supported
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
        'bundle': object,
        'name': str,
        'parnamefmt': str, # parameter name format, supports: {i} and {name}
        'count': int,
        'central': float,
        'uncertainty': float,
        Optional('unctype',     default='absolute'): Or('absolute', 'relative', 'percent'),
        Optional('label',       default='VarArray'): str,
        Optional('parlabelfmt', default='array {i}'): str,
        Optional('edges'):   Or((), [], np.ndarray), # passed to the parlabelfmt as relevant {left}, {right} and {width}
        Optional('centers'): Or((), [], np.ndarray), # passed to the parlabelfmt as relevant {center}
        Optional('fixed', default=False): bool,
        })

    @staticmethod
    def _provides(cfg):
        name = cfg['name']
        return (), (name,)

    def build(self):
        name = self.vcfg['name']
        label = self.vcfg['label']
        self.vararray = C.VarArray(self.parnames, labels=label)
        self.set_output(name, None, self.vararray.vararray.points)

    def define_variables(self):
        name = self.vcfg['name']
        parnamefmt = self.vcfg['parnamefmt']
        parlabelfmt = self.vcfg['parlabelfmt']
        count = self.vcfg['count']

        edges = self.vcfg['edges']
        if len(edges):
            edges = np.asanyarray(edges)
            centers = 0.5*(edges[1:]+edges[:-1])
        else:
            centers = self.vcfg['centers']

        assert len(edges)-1==count, 'Number of edges and number of elements are not consistent'
        assert len(centers)==count, 'Number of centers and number of elements are not consistent'

        central = self.vcfg['central']
        uncertainty = self.vcfg['uncertainty']
        unctype = self.vcfg['unctype']
        if unctype=='absolute':
            sigma = uncertainty
        elif unctype=='relative':
            sigma = central*uncertainty
        elif unctype=='percent':
            sigma = central*(uncertainty*0.01)

        self.parameters=[]
        self.parnames=list(parnamefmt.format(i=i, name=name) for i in range(count))
        fixed = self.vcfg['fixed']
        for i, (parname, left, right, center) in enumerate(it.zip_longest(self.parnames, edges[:-1], edges[1:], centers)):
            label = parlabelfmt.format(i=i, left=left, right=right, center=center, width=right-left)

            par = self.reqparameter(parname, None, central=central, sigma=sigma, label=label, fixed=fixed)
            self.parameters.append(par)
