from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or

class trans_summatordiag_v01(TransformationBundle):
    """SumMatOrDiag bundle

    Supports:
        - no major indices
        - multiple instances

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
        - ninputs     - integer
        """
    _validator = Schema({
            'bundle': object,
            'instances': {str: Or(str, None)},
            'ninputs': int,
        })

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    @staticmethod
    def _provides(cfg):
        return (), cfg.instances.keys()

    def build(self):
        self.objects = []
        instances = self.vcfg['instances']
        ninputs = self.vcfg['ninputs']

        for name, label in instances.items():
            if label is None:
                label = 'SumMatOrDiag {autoindex}'

            for it in self.nidx_minor.iterate():
                sum_mat_diag = C.SumMatOrDiag(labels=it.current_format(label))
                self.objects.append(sum_mat_diag)

                for i in range(ninputs):
                    inp = sum_mat_diag.add_input('input_{:02d}'.format(i))
                    self.set_input(name, it, inp, argument_number=i)

                self.set_output(name, it, sum_mat_diag.sum)


