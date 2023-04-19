from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle
from schema import Schema, Or, Optional, Use, And
from gna.configurator import StripNestedDict
import inspect

strategies = {
        'constant':    R.GNA.Interpolation.Strategy.Constant,
        'extrapolate': R.GNA.Interpolation.Strategy.Extrapolate,
        'nearestedge': R.GNA.Interpolation.Strategy.NearestEdge,
        }
def ispair(lst):
    return len(lst)==2
IsStrategy=Or(*strategies.keys())
IsStrategy2=And(Or((IsStrategy,), [IsStrategy]), ispair)

class interpolation_1d_v01(TransformationBundle):
    '''Create interpolation graph

    Configuration options:
    - name - output names
    '''
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    _validator = Schema({
            'bundle': object,
            'name': str,
            'kind': Or('linear', 'expo', 'log', 'logx', 'const'),
            Optional('strategy'): Or(IsStrategy2, And(IsStrategy, Use(lambda x: [x, x]))),
            Optional('label', default=''): str,
            Optional('labelfmt', default=''): str
        })

    @staticmethod
    def _provides(cfg):
        name = cfg['name']
        return (), (name, f'{name}_base')

    def build(self):
        name = self.vcfg['name']
        kind = self.vcfg['kind']
        label= self.vcfg['label']
        labelfmt = self.vcfg['labelfmt']
        InterpClass = getattr(R, f'Interp{kind.capitalize()}')

        self.interpolators=[]
        def makeinterp(label):
            interp = InterpClass(labels=label)
            self.interpolators+=interp,

            try:
                under, over=self.vcfg['strategy']
            except KeyError:
                pass
            else:
                interp.set_underflow_strategy(strategies[under])
                interp.set_overflow_strategy(strategies[over])

            return interp

        for imajor, itmajor in enumerate(self.nidx_major):
            interp = makeinterp(itmajor.current_format(label))
            sampler = interp.transformations.front()
            inp_coarse = [sampler.inputs.edges]
            inp_fine = [sampler.inputs.points]

            if imajor:
                trans = interp.add_transformation()
            else:
                trans = interp.transformations.back()
            trans.setLabel(itmajor.current_format(labelfmt))
            inp_fine.append(trans.inputs.newx)

            inp_coarse+=trans.inputs.x,
            inp_fine+=trans.inputs.newx,

            for iminor, itminor in enumerate(self.nidx_minor):
                it = itmajor+itminor
                inp = interp.add_input()

                self.set_input(name,  it, inp, argument_number=0)
                self.set_output(name, it, trans.outputs.back())

            self.set_input(f'{name}_base', itmajor, inp_coarse, argument_number=0)
            self.set_input(f'{name}_base', itmajor, inp_fine, argument_number=1)

