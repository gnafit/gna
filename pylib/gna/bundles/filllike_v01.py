from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, Use, And

class filllike_v01(TransformationBundle):
    """FillLike bundle

    Supports:
        - no major indices
        - multiple instances

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
        - value       - float value
        """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
            'bundle': object,
            'instances': {str: Or(str, None)},
            'value': Or(And(int, Use(float)), float),
        })

    @staticmethod
    def _provides(cfg):
        return (), cfg.instances.keys()

    def build(self):
        self.objects = []
        instances = self.vcfg['instances']
        value = self.vcfg['value']
        for name, label in instances.items():
            if label is None:
                label = f'FillLike {value}|{{autoindex}}'

            for it in self.nidx_minor:
                fl = C.FillLike(value, labels=it.current_format('{{'+label+'}}'))
                self.objects.append(fl)

                self.set_input(name, it, fl.fill.inputs.a, argument_number=0)
                self.set_output(name, it, fl.fill.outputs.a)
