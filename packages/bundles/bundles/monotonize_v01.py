from gna import constructors as C
from gna.configurator import StripNestedDict
from tools.schema import Schema, Optional, And

from gna.bundle import TransformationBundle

class monotonize_v01(TransformationBundle):
    """Make Monotonize instance

    Configuration option:
        - instances       - dict[str, Optional[str]], key - name of transformation, value - label
    """

    _validator = Schema(
        {
            "bundle": object,
            "instances": {str: str},
            Optional('gradient', default=0.0): float,
            Optional('index_fraction', default=0.0): And(float, lambda f: 0<=f<1),
        }
    )

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    @classmethod
    def _provides(cls, cfg):
        vcfg = cls._validator.validate(StripNestedDict(cfg))
        return (), tuple(vcfg["instances"])

    def build(self):
        self.objects = []
        index_fraction = self.vcfg['index_fraction']
        gradient = self.vcfg['gradient']
        for name, label in self.vcfg["instances"].items():
            for it in self.nidx.iterate():
                monotonize = C.Monotonize(index_fraction, gradient)
                self.objects.append(monotonize)

                if not label:
                    label = "Monotonize\n{autoindex}"

                trans = monotonize.monotonize
                trans.setLabel(label)

                self.set_input(name, it, trans.x, argument_number=0)
                self.set_input(name, it, trans.y,  argument_number=1)
                self.set_output(name, it, trans.yout)
