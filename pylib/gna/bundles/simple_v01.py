from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class simple_v01(TransformationBundle):
    """Simple bundle, that is using methods, provided via configuration in order to announce inputs/outputs"""
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        actions = self.cfg.actions
        obj_action = actions.object
        input_action = actions.get('input')
        output_action = actions.output

        self.obj = obj_action()
        name = self.cfg.name
        for it in self.nidx_minor.iterate():
            if input_action:
                inp = input_action(self.obj)
                self.set_input(name, it, inp, argument_number=0)

            out = output_action(self.obj)
            self.set_output(name, it, out)


