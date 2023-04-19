from load import ROOT as R
import numpy as np

from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import Schema, Optional, And


class tao_fastn_spectrum_v01(TransformationBundle):
    """The implementation of TAO fast neutron background based on
    f63dd5ffd5fff22f3feb2b::packages/dayabay/bundles/dayabay_fastn_power_v02.py

    Configuration option:
        - name                - str, name of spectrum in equtions and ns for parameters
        - parameters          - dict[str, tuple], dict of parameters $A * \exp(B*x) + C$
        - label               - str, label of data
        - order               - int, order of GL integration
        - edges               - array, edges of intemgration
        - normalize           - bool, normalize hist on edges
        - normalization_range - optional (float, float) normalization range
    """
    _validator = Schema(And({
                'bundle': object,
                'name': str,
                'parameters': dict,
                'label': str,
                'order': int,
                'edges': np.ndarray,
                'normalize': bool,
                Optional('normalization_range', default=()): tuple,
            },
            )
        )
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

        self.objects=[]
        self.vcfg: dict = self._validator.validate(StripNestedDict(self.cfg))

        self._ns = self.namespace(self.vcfg["name"])

    @classmethod
    def _provides(cls, cfg):
        cfg: dict = cls._validator.validate(StripNestedDict(cfg))
        return (), (cfg["name"],)

    def _normalize(self, hist, it):
        edges = self.vcfg["edges"]
        normalization_range = self.vcfg.get('normalization_range')
        if normalization_range:
            emin, emax = normalization_range
            try:
                ((imin, imax),) = np.where(((edges - emin) * (edges - emax)).round(6) == 0.0)
            except ValueError:
                raise ValueError(f"Cannot determine normalization region for Fast Neutrons ({emin}, {emax})")

            normalize = R.Normalize(int(imin), int(imax - imin))
        else:
            normalize = R.Normalize()

        hist.outputs.back() >> normalize.normalize
        normalize.normalize.setLabel(it.current_format("Fast neutron hist {site} (norm)"))

        return normalize

    def build(self):
        edges = self.vcfg["edges"]

        with self._ns:
            integrator_gl = R.IntegratorGL(edges.shape[0] - 1, self.vcfg["order"], edges, ns=self.namespace)
            integrator_gl.points.setLabel("Fast neutron energy points")
            bins = integrator_gl.points.x

            exp_arg = R.WeightedSum(["B"], [bins])
            exp_t = R.Exp()
            exp_t.single_input().connect(exp_arg)

            fill_t = R.FillLike(1.)
            fill_t.single_input().connect(bins)

            func = R.WeightedSum(["A", "C"], [exp_t.single(), fill_t.single()])
            func.sum.setLabel("Fast neutron shape")

            hist = integrator_gl.hist
            hist.setLabel("Fast neutron hist")

            func.sum >> hist.f

        target = hist.single()
        for it in self.nidx:
            norm=None
            if self.vcfg["normalize"]:
                norm = self._normalize(hist, it)
                target = norm.single()

            self.set_output(self.vcfg["name"], it, target)

            # """Provide the outputs and objects"""
            self.context.objects[("integrator",) + it.current_values()] = integrator_gl
            self.context.objects[("func",) + it.current_values()] = func
            self.context.objects[("hist",) + it.current_values()] = hist
            if norm:
                self.context.objects[("normalize",) + it.current_values()] = norm

    def define_variables(self):
        for par, (central, sigma) in self.vcfg["parameters"].items():
            self._ns.reqparameter(par, central=central, sigma=sigma, fixed=False)
