
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class predefined_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        # Configuration:
        #   name    = string
        #   outputs = NestedDict()
        #   inputs  = NestedDict()

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        name = self.cfg.name
        for it_major in self.nidx_major:
            path = it_major.current_values()

            output = None if self.cfg.outputs is None else self.cfg[('outputs',)+path]
            inputs = ()   if self.cfg.inputs  is None else self.cfg[('inputs',) +path]

            for it_minor in self.nidx_minor:
                it = it_major + it_minor

                if output:
                    self.set_output(name, it, output)

                for arg, input in enumerate(inputs):
                    self.set_input(name, it, input, argument_number=arg)
