from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, Use, And

class conditional_product_v01(TransformationBundle):
    """Conditional product bundle

    Supports:
        - no major indices
        - same condition for all minor indices
        - multiple instances

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
        - nprod[=1]   - number of elements to multiply when condition is 0
        - ninputs[=2] - number of inputs to multiply
        - default[=1.0] - default value for the condition
        - condlabel[=''] - default label for the condition

    Predefined names:
        - 'condition' - variable for condition

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
            Optional('nprod', default=1): int,
            Optional('ninputs', default=2): int,
            Optional('default', default=1.0): And(Or(int, float), Use(float)),
            Optional('condlabel', default='Condiction for a product'): str,
        })

    @staticmethod
    def _provides(cfg):
        return ('condition',), cfg.instances.keys()

    def build(self):
        self.objects = []
        instances = self.vcfg['instances']
        nprod   = self.vcfg['nprod']
        ninputs = self.vcfg['ninputs']
        for name, label in instances.items():
            if label is None:
                label = 'Conditional product | {autoindex}'

            for it in self.nidx_minor.iterate():
                cprod = C.ConditionalProduct(nprod, self.varname, labels=it.current_format(label))
                self.objects.append(cprod)

                for i in range(ninputs):
                    iname='input_{:02d}'.format(i)
                    self.set_input(name, it, cprod.add_input(iname), argument_number=i)

                self.set_output(name, it, cprod.product.product)

    def define_variables(self):
        label   = self.vcfg['condlabel']
        central = self.vcfg['default']
        self.reqparameter(self.varname, None, central=central, fixed=True, label=label)

