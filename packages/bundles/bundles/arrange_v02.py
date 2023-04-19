import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import *

class arrange_v02(TransformationBundle):
    """Bundle combines multiple arguments as a single Transformation with multiple indices

       The connection is done via View tranformation, which does not allocate extra memory
       and its core function is empty.

       Expects:
           - single major index
           - no variables

        Changes since arrange v01:
            + enable minor indices
            + add a way to combine inputs

        For configuration with name=[trans, ...] provides:
            - [out] trans
            - [in] trans.01, trans.02, ... for the number of major indices: connect outputs to outputs
            - [in] trans_in01.01, trans_in02.01 for the number of major indices: connect inputs to inputs

        Usage in the expression:
            - name='trans'
            - In case there are no inputs:
                + 'combine_trans[m](out1, out2, ...)' - binding
                + 'trans()' - subsequentusage
            - In case there are inputs:
                + 'combine_trans[m](out0(combine_trans_it00_in00(), combine_trans_it00_in01()), out1(out1| combine_trans_it01_in00(), combine_trans_it01_in01(), ...), ...)'
                + 'trans| out_for_trans()'
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
        'bundle': object,
        'names': Or([str], And(str, Use(lambda s: [s]))),
        Optional('ninputs', default=0): int,               # number of inputs for each element
        Optional('nmajor', default=0): int                 # number of major entries
        })

    @classmethod
    def _provides(cls, cfg) -> tuple[tuple[str, ...], tuple[str, ...]]:
        vcfg = cls._validator.validate(StripNestedDict(cfg))
        names = list(vcfg['names'])
        names_out = list(cls._format_combine(name) for name in names)

        nmajor = vcfg['nmajor']
        ninputs = vcfg['ninputs']
        names_in  = list(cls._format_input(name, imajor, iinput) for name in names
                                                                 for imajor in range(nmajor)
                                                                 for iinput in range(ninputs)
                                                                 )
        return (), tuple(names + names_out + names_in)

    @staticmethod
    def _format_combine(name: str) -> str:
        return f'combine_{name}'

    @staticmethod
    def _format_input(name: str, imajor: int, iinput: int) -> str:
        return arrange_v02._format_combine(name)+f'_it{imajor:02d}_in{iinput:02d}'

    def build(self) -> None:
        names = self.vcfg['names']
        objects = self.objects = {name: [] for name in names}
        def newview(name, label):
            v = C.View(labels=label)
            objects[name].append(v)
            return v

        ninputs = self.vcfg['ninputs']
        n = self.nidx_major.get_size()
        for imajor, itmajor in enumerate(self.nidx_major.iterate()):
            for itminor in self.nidx_minor.iterate():
                it = itminor+itmajor
                namemajor = itmajor.current_format()
                for name in names:
                    #
                    # Combining transformation: create inputs for extrenal outputs
                    #
                    cname = self._format_combine(name)
                    viewout = newview(name, f'{name} out [{namemajor}]')
                    transout = viewout.view
                    for j in range(n):
                        if imajor==j:
                            self.set_input(cname, it, transout.inputs.data, argument_number=imajor)
                        else:
                            self.set_input(cname, it, (), argument_number=j)

                    #
                    # Combining transformation: create outputs for extrenal inputs
                    #
                    for iinput in range(ninputs):
                        viewin = newview(name, f'{name} in {iinput} [{namemajor}]')
                        inname = self._format_input(name, imajor, iinput)

                        transin = viewin.view
                        self.set_output(inname, itminor, transin.outputs.view)
                        self.set_input(name, it, transin.inputs.data, argument_number=iinput)

                    #
                    # Combined transformation: pass outputs and inputs
                    #
                    self.set_output(name, it, transout.view)

