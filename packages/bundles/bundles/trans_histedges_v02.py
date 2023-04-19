from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle

class trans_histedges_v02(TransformationBundle):
    """HistEdges bundle: defines HistEdges object with edges, bin centers and bin widths

    Changes since v01:
        - [fix] Each instance gets only one input
    """
    types = set(('edges', 'centers', 'widths'))
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

    @classmethod
    def _provides(cls, cfg):
        types = cfg.get('types')
        types = cls.types if types is None else cls.types.intersection(types)
        names = tuple(cfg.instances.keys())
        outputs = tuple('{}_{}'.format(name, t) for name in names for t in types)
        return (), names+outputs

    def build(self):
        self.objects = []
        types = self.cfg.get('types')
        types = self.types if types is None else self.types.intersection(types)
        for name, label in self.cfg.instances.items():
            if label is None:
                label = 'HistEdges {autoindex}'

            for it in self.nidx_minor.iterate():
                histedges = C.HistEdges(labels=it.current_format(label))
                self.objects.append(histedges)
                trans = histedges.histedges

                self.set_input(name, it, trans.hist, argument_number=0)
                for t in types:
                    cname = '{}_{}'.format(name, t)
                    self.set_output(cname, it, trans.outputs[t])


