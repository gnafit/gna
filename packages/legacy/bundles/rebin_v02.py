from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class rebin_v02(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        super(rebin_v02, self).__init__( *args, **kwargs )
        self.init_indices()

    def build(self):
        for it in self.idx.iterate():
            rebin = self.objects[('rebin',)+it.current_values()] = C.Rebin( self.cfg.edges, self.cfg.rounding)

            rebin.rebin.setLabel(it.current_format('Rebin\n{autoindex}'))
            self.set_input(rebin.rebin.histin, 'rebin', it, clone=0)
            self.set_output(rebin.rebin.histout, 'rebin', it)


