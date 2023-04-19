from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class rebin_v05(TransformationBundle):
    """Rebin bundle v05
    Defines multiple rebin transformations.

    Changes since v04:
      - Provide a histogram and points with binning.
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @staticmethod
    def _provides(cfg):
        return (), list(cfg.instances.keys())+['rebin_hist', 'rebin_points']

    def build(self):
        edges = self.cfg.edges
        hist = C.Histogram(edges, labels='Rebin edges spec')
        points = C.Points(edges, labels='Rebin edges spec')
        self.set_output('rebin_hist', None, hist.hist.hist)
        self.set_output('rebin_points', None, points.points.points)

        self.objects = [hist, points]
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'Rebin {autoindex}'

            for it in self.nidx_minor.iterate():
                rebin = C.Rebin(edges, self.cfg.rounding, labels=it.current_format(label))
                self.objects.append(rebin)

                self.set_input( name, it, rebin.rebin.histin, argument_number=0)
                self.set_output(name, it, rebin.rebin.histout)
