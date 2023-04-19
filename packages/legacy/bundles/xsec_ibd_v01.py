from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class xsec_ibd_v01(TransformationBundleLegacy):
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
    #       # this line says that the bundle will provide the 'ee' - positron energy as additional output.
    #       provides = [ 'ibd_xsec', 'ee', 'enu' ]
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
    #       # this line says that the bundle will provide the 'ee' - positron energy as additional output.
    #       provides = [ 'ibd_xsec', 'ee', 'enu', 'jacobian' ]
    #       )
    # @endcode
    def __init__(self, *args, **kwargs):
        TransformationBundleLegacy.__init__( self, *args, **kwargs )

        # check that order is sane
        if not self.cfg.order in [0, 1]:
            raise Exception("Unsupported ibe order {} (should be 0 or 1)".format(self.cfg.order))

    def build(self):
        # initalize Evis to Enu converter
        with self.common_namespace("ibd"):
            self.econv = R.EvisToEe()

        # register it's input and output
        self.set_input( self.econv.Ee.Evis, 'ee', clone=0 )
        self.set_output( self.econv.Ee.Ee,  'ee' )

        if self.cfg.order==0:
            # in 0th order and 1d case
            # create 1d cross section
            with self.common_namespace("ibd"):
                self.ibd = R.IbdZeroOrder()

            # register Enu input and output
            self.set_input(self.ibd.Enu.Ee,   'enu', clone=0)
            self.set_output(self.ibd.Enu.Enu, 'enu')

            # register cross section input and output
            self.set_input(self.ibd.xsec.Ee,    'ibd_xsec', clone=0)
            self.set_output(self.ibd.xsec.xsec, 'ibd_xsec')

            #label cross section for the graph
            self.ibd.xsec.setLabel('IBD xsec (0)')
        elif self.cfg.order==1:
            # in 1th order and 2d case
            # create 2d cross section
            with self.common_namespace("ibd"):
                self.ibd = R.IbdFirstOrder()

            # register Enu inputs and output
            self.set_input(self.ibd.Enu.Ee,     'enu', clone=0)
            self.set_input(self.ibd.Enu.ctheta, 'enu', clone=1)
            self.set_output(self.ibd.Enu.Enu,   'enu')

            # register cross section inputs and output
            self.set_input(self.ibd.xsec.Enu,    'ibd_xsec', clone=0)
            self.set_input(self.ibd.xsec.ctheta, 'ibd_xsec', clone=1)
            self.set_output(self.ibd.xsec.xsec,  'ibd_xsec')
            # label cross section
            self.ibd.xsec.setLabel('IBD xsec (1)')

            # register jacobian inputs and output
            self.set_input(self.ibd.jacobian.Enu,       'jacobian', clone=0)
            self.set_input(self.ibd.jacobian.Ee,        'jacobian', clone=1)
            self.set_input(self.ibd.jacobian.ctheta,    'jacobian', clone=2)
            self.set_output(self.ibd.jacobian.jacobian, 'jacobian')
            # label jacobian
            self.ibd.jacobian.setLabel('Ee->Enu jacobian')

        # label neutrino energy caclulator
        self.ibd.Enu.setLabel('Enu')

    def define_variables(self):
        # initialize necessary variables: neutron lifetime and mass (proton, neutron, electron)
        # in common namespace
        from gna.parameters import ibd
        ibd.reqparameters(self.common_namespace('ibd'))

