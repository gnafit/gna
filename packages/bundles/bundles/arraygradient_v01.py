from gna import constructors as C
from gna.configurator import StripNestedDict
from tools.schema import Schema, Optional, And, Or

from gna.bundle import TransformationBundle

class arraygradient_v01(TransformationBundle):
    """Make ArrayGradient2p/ArrayGradient3p instance

    Configuration option:
        - instances - dict[str, Optional[str]], key - name of transformation, value - label
        - npoints - number of points to use to build gradient:
            - 2 - ArrayGradient2p, use 2 points, x_i=center(x_i, x_i+1)
            - 3 - ArrayGradient3p, use 3 points, x_i=x_i
    """

    _validator = Schema(
        {
            "bundle": object,
            "instances": {str: str},
            "npoints": Or(2, 3)
        }
    )

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    @classmethod
    def _provides(cls, cfg):
        vcfg = cls._validator.validate(StripNestedDict(cfg))
        names = list(vcfg["instances"])
        return (), tuple(names + [n+'_x' for n in names])

    def build(self):
        self.objects = []
        npoints = self.vcfg['npoints']
        GradientClass = npoints==2 and C.ArrayGradient2p or C.ArrayGradient3p

        for name, label in self.vcfg["instances"].items():
            for it in self.nidx.iterate():
                grad = GradientClass()
                self.objects.append(grad)

                if not label:
                    label = "Gradient\n{autoindex}"

                trans = grad.gradient
                trans.setLabel(label)

                self.set_input(name, it, trans.x, argument_number=0)
                self.set_input(name, it, trans.y,  argument_number=1)
                self.set_output(name, it, trans.gradient)

                if npoints==2:
                    self.set_output(name+'_x', it, trans.xout)
