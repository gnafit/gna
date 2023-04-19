import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from typing import Tuple
from tools.schema import *

class concatenate_v01(TransformationBundle):
    """Concatenates arguments
       Usage in the expression:
           - instances: {name: (ninputs, label)} combinations
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
        'bundle': object,
        'instances': {str: (int, str)},
        })

    @classmethod
    def _provides(cls, cfg) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (), tuple(cfg['instances'].keys())

    def build(self) -> None:
        objects = self.objects = []

        for name, (ninputs, labelfmt) in self.vcfg['instances'].items():
            if labelfmt is None:
                labelfmt = 'Concat '+name+' {autoindex}'

            for it in self.nidx.iterate():
                concat = C.Concat(labels=it.current_format(labelfmt))
                objects.append(concat)

                for iinput in range(ninputs):
                    input = concat.add_input(f'element_{iinput:02d}')
                    self.set_input(name, it, input, argument_number=iinput)

                self.set_output(name, it, concat.concat.concat)

