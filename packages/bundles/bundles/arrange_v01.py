import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import *

class arrange_v01(TransformationBundle):
    """Bundle combines multiple arguments as a single Transformation with multiple indices

       The connection is done via View tranformation, which does not allocate extra memory
       and its core function is empty.

       Expects:
         - single major index
         - no other implicit indices
         - no variables
    """

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
        'bundle': object,
        'names': Or([str], And(str, Use(lambda s: [s]))),
        # Optional('verbose', default=0), Or(int, And(bool, Use(int)))
        })

    @classmethod
    def _provides(cls, cfg):
        vcfg = cls._validator.validate(StripNestedDict(cfg))
        return (), tuple(vcfg['names'])

    def build(self):
        names = self.vcfg['names']
        objects = self.objects = {name: [] for name in names}
        def newview(name, label):
            v = C.View(labels=label)
            objects[name].append(v)
            return v

        n = self.nidx_major.get_size()
        for i, it in enumerate(self.nidx_major.iterate()):
            for name in names:
                itname = it.current_format()
                v = newview(name, f'{name} [{itname}]')

                trans = v.view
                self.set_input(name, it, trans.inputs.data, argument_number=i)
                self.set_output(name, it, trans.view)

                for j in range(n):
                    if i!=j:
                        self.set_input(name, it, (), argument_number=j)

