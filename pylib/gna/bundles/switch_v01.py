from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, Use, And

class switch_v01(TransformationBundle):
    """Switch bundle

    Supports:
        - no major indices
        - same switch for all minor indices
        - multiple instances

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
        - ninputs[=2] - number of inputs to use
        - varlabel[=''] - default label for the condition variable

    Predefined names:
        - 'switch' - variable for condition

        (may be configured via 'names' option of a bundle)
        """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

        self.varname = self.get_globalname('condition')

    _validator = Schema({
            'bundle': object,
            'instances': {str: Or(str, None)},
            Optional('default', default=0): int,
            Optional('ninputs', default=2): int,
            Optional('varlabel', default='Switch variable'): str,
        })

    @staticmethod
    def _provides(cfg):
        return ('switch',), cfg.instances.keys()

    def build(self):
        self.objects = []
        instances = self.vcfg['instances']
        ninputs = self.vcfg['ninputs']
        for name, label in instances.items():
            if label is None:
                label = 'Switch | {autoindex}'

            for it in self.nidx_minor.iterate():
                cswitch = C.Switch(self.varname, labels=it.current_format(label))
                self.objects.append(cswitch)

                for i in range(ninputs):
                    iname='input_{:02d}'.format(i)
                    self.set_input(name, it, cswitch.add_input(iname), argument_number=i)

                self.set_output(name, it, cswitch.switch.result)

    def define_variables(self):
        label = self.vcfg['varlabel']
        value = self.vcfg['default']
        self.reqparameter(self.varname, None, central=value, fixed=True, label=label)

