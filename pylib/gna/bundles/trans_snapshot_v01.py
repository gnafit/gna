
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle

class trans_snapshot_v01(TransformationBundle):
    """Snapshot: implements snapshot bundle"""
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @staticmethod
    def _provides(cfg):
        return (), cfg.instances.keys()

    def build(self):
        self.objects = []
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'Snapshot {autoindex}'

            for it in self.nidx_minor.iterate():
                snapshot = C.Snapshot(labels=it.current_format(label))
                self.objects.append(snapshot)

                self.set_input( name, it, snapshot.snapshot.source, argument_number=0)
                self.set_output(name, it, snapshot.snapshot.result)


