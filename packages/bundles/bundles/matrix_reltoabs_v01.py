from gna import constructors as C
from gna.configurator import StripNestedDict
from tools.schema import Schema

from gna.bundle import TransformationBundle


class matrix_reltoabs_v01(TransformationBundle):
    """Make

    Configuration option:
        - instances       - dict[str, Optional[str]], key - name of transformation, value - label
    """

    _validator = Schema(
        {
            "bundle": object,
            "instances": {str: str},
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
        for name, label in self.vcfg["instances"].items():
            for it in self.nidx.iterate():
                matrecovery = C.MatrixRelToAbs()
                self.objects.append(matrecovery)

                if not label:
                    label = "Matrix abs from rel\n{autoindex}"

                trans = matrecovery.product
                trans.setLabel(label)

                self.set_input(name, it, trans.spectra, argument_number=0)
                self.set_input(name, it, trans.matrix, argument_number=1)
                self.set_output(name, it, trans.product)
