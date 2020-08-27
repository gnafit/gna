# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

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

        self.varname = self.get_globalname('condition')

    @staticmethod
    def _provides(cfg):
        return ('condition',), cfg.instances.keys()

    def build(self):
        self.objects = []
        instances = self.cfg['instances']
        nprod   = self.cfg.get('nprod', 1)
        ninputs = self.cfg.get('ninputs', 2)
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
        label   = self.cfg.get('condlabel', 'Condiction for a product')
        central = self.cfg.get('default', 1.0)
        self.reqparameter(self.varname, None, central=central, fixed=True, label=label)

