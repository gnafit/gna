from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from tools.schema import Schema, Optional
from typing import Tuple, Dict, Any
from gna.configurator import StripNestedDict

class rebin_input_v01(TransformationBundle):
    """Rebin (RebinInput) bundle v01
    Defines multiple rebin transformations with common binning, provided via input.

    Changes since rebin_v05:
      - Switch to RebinInput
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.vcfg: Dict[str, Any] = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
            'bundle': object,
            'instances': {str: str},
            'rounding': int,
            Optional('matrix_label', default='Rebin matrix'): str,
            Optional('expose_matrix', default=False): bool,
            Optional('dynamic_edges', default=False): bool,
            Optional('permit_underflow', default=False): bool
        })

    @staticmethod
    def _provides(cfg) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        instances = tuple(cfg.instances.keys())
        return (), ('rebin_edges',)+instances

    def build(self) -> None:
        expose_matrix = R.GNA.DataPropagation.Propagate if self.vcfg['expose_matrix'] else R.GNA.DataPropagation.Ignore
        edges_mode    = R.GNA.DataMutability.Dynamic    if self.vcfg['dynamic_edges'] else R.GNA.DataMutability.Static

        rebin = C.RebinInput(self.vcfg['rounding'], edges_mode, expose_matrix, labels=self.vcfg['matrix_label'])
        if self.vcfg['permit_underflow']:
            rebin.permitUnderflow()

        self.set_input('rebin_edges', None, rebin.matrix.HistEdgesOut, argument_number=0)
        self.set_input('rebin_edges', None, rebin.matrix.EdgesIn, argument_number=1)
        self.objects = [rebin]
        count=0
        for name, label in self.cfg.instances.items():
            if label is None:
                label = '%s {autoindex}'%name

            for it in self.nidx_minor.iterate():
                if count>0:
                    rebin.add_transformation()
                    rebin.add_input()
                count+=1
                trans=rebin.transformations.back()
                trans.setLabel(it.current_format(label))

                self.set_input( name, it, trans.histin, argument_number=0)
                self.set_output(name, it, trans.histout)
