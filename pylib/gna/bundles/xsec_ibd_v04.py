from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.context import entryContext
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, Use, And

class xsec_ibd_v04(TransformationBundle):
    """Inverse beta-decay cross section from Vogel&Beacom v04
    Changes since xsec_ibd_v03:
        - Added IbdFirst order constructor arguments
        - Add validator
    """
    ## @brief Inverse Beta Decay (IBD) cross section by Vogel&Beacom (arxiv: 9903.554)
    #
    # IBD: bar(νe) + p → n + e+
    #
    # IBD cross section is available of two orders:
    #   - 0th order (1d: Enu)
    #   - 1st order (2d: Enu, cosθ)
    #
    # Since the positron created in the IBD usually annihilates the bundle also provides the conversion betwee
    # positron energy Ee and visible energy Evis=(Ee+me).
    #
    # When build for the 0th order the cross section is 1d function of positron energy (Ee). The bundle provides:
    #  - 'ibd_xsec' — the IBD cross section as a function of Ee
    #  - 'enu' — the neutrino energy as a function of positron energy Ee
    #  - 'ee' — the positron energy as a function of 'visible' energy Evis=(Ee+me).
    #  In order to work the variables should be connected: evis->ee->enu and ee->ibd_xsec.
    #
    # When built for the 1th order the cross section is a 2d function of neutrino energy (Enu) and positron angle (cosθ).
    # The bundle provides:
    #  - 'ibd_xsec' — the IBD cross section as a function of Enu and positron angle cosθ
    #  - 'enu' — the neutrino energy as a function of positron energy Ee and positron angle cosθ
    #  - 'ee' — the positron energy as a function of 'visible' energy Evis=(Ee+me).
    #  - 'jacobian' — jacobian(Ee, Enu, cosθ) for transition from Evis/Ee to Enu.
    #  In order to work the variables should be connected as follows:
    #  - evis->ee
    #  - ee, enu and cosθ->jacobian
    #  - enu and coθ->ibd_xsec.
    #
    # Configuration for 0th order:
    # @code{.py}
    #   ibd_xsec = NestedDict(
    #       # bundle name to be executed
    #       bundle = 'xsec_ibd_v01',
    #       # and its parameters:
    #       # - the IBD cross section order (0 for zero-th or 1 the first). First is not yet implemented.
    #       order = 0,
    #       )
    #
    # Configuration for 1th order:
    # @code{.py}
    #   ibd_xsec = NestedDict(
    #       # bundle name to be executed
    #       bundle = 'xsec_ibd_v01',
    #       # and its parameters:
    #       # - the IBD cross section order (0 for zero-th or 1 the first). First is not yet implemented.
    #       order = 1,
    #       )
    # @endcode
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        constants = self.vcfg.get('constants')
        if constants:
            self._arguments = tuple(constants[k] for k in ('PhaseFactor', 'g', 'f', 'f2'))
        else:
            self._arguments = ()

    _validator = Schema({
            'bundle': object,
            'order': Or(0, 1),
            Optional('constants'): {
                'PhaseFactor': float,
                'g': float,
                Optional('f', default=1): float,
                'f2': float
                },
            Optional('verbose', default=False): Or(bool, Use(int))
        })

    @staticmethod
    def _provides(cfg):
        if cfg.order:
            return (), ('ibd_xsec', 'ee', 'enu', 'jacobian', 'enuToEvis0')
        else:
            return (), ('ibd_xsec', 'ee', 'enu')

    def build(self):
        order = self.vcfg['order']
        if self.vcfg['verbose']:
            print(f'Initializing {order} order IBD cross section', end='')
            constants = self.vcfg['constants']
            print(f'with {constants!s}') if constants else print()

        with entryContext(subgraph="IBD"):
            # initalize Evis to Enu converter
            # with self.namespace("ibd"):
            self.econv = R.EvisToEe()

            # register it's input and output
            self.set_input('ee', None, self.econv.Ee.Evis, argument_number=0)
            self.set_output('ee', None, self.econv.Ee.Ee)

            if order==0:
                # in 0th order and 1d case
                # create 1d cross section
                # with self.namespace("ibd"):
                self.ibd = R.IbdZeroOrder(*self._arguments)

                # register Enu input and output
                self.set_input('enu', None, self.ibd.Enu.Ee, argument_number=0)
                self.set_output('enu', None, self.ibd.Enu.Enu)

                # register cross section input and output
                self.set_input('ibd_xsec', None, self.ibd.xsec.Ee, argument_number=0)
                self.set_output('ibd_xsec', None, self.ibd.xsec.xsec)

                #label cross section for the graph
                self.ibd.xsec.setLabel('IBD xsec (0)')
            elif order==1:
                # in 1th order and 2d case
                # create 2d cross section
                # with self.namespace("ibd"):
                self.ibd = R.IbdFirstOrder(*self._arguments)

                # register Enu inputs and output
                self.set_input('enu', None, self.ibd.Enu.Ee, argument_number=0)
                self.set_input('enu', None, self.ibd.Enu.ctheta, argument_number=1)
                self.set_output('enu', None, self.ibd.Enu.Enu)

                # register cross section inputs and output
                self.set_input('ibd_xsec', None, self.ibd.xsec.Enu, argument_number=0)
                self.set_input('ibd_xsec', None, self.ibd.xsec.ctheta, argument_number=1)
                self.set_output('ibd_xsec', None, self.ibd.xsec.xsec)
                # label cross section
                self.ibd.xsec.setLabel('IBD xsec (1)')

                # register jacobian inputs and output
                self.set_input('jacobian', None, self.ibd.jacobian.Enu, argument_number=0)
                self.set_input('jacobian', None, self.ibd.jacobian.Ee, argument_number=1)
                self.set_input('jacobian', None, self.ibd.jacobian.ctheta, argument_number=2)
                self.set_output('jacobian', None, self.ibd.jacobian.jacobian)
                # label jacobian
                self.ibd.jacobian.setLabel('Ee-\\>Enu jacobian')

            # label neutrino energy caclulator
            self.ibd.Enu.setLabel('Enu')

        if self.vcfg['verbose']>1:
            print('Dump cross section:')
            self.ibd.dump()

    def define_variables(self):
        pass
