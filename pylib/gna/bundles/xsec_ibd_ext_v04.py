from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.context import entryContext
from tools.root_helpers import TFileContext

class xsec_ibd_ext_v04(TransformationBundle):
    """Inverse beta-decay cross section from Vogel&Beacom
    Changes since xsec_ibd_v03:
        - Add an option to load external ROOT file with TF2 cross section
        - 0th order is disabled

    Configuration:
    ibd_xsec = NestedDict(
        bundle = 'xsec_ibd_v03',
        filename = 'path/to/file.root',
        order = 1,
        objectname = 'objectname',
        )
    """
    ## @brief Inverse Beta Decay (IBD) cross section by Vogel&Beacom (arxiv: 9903.554)
    #
    # IBD: bar(νe) + p → n + e+
    #
    # IBD cross section is available of two orders:
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
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

        # check that order is sane
        if not self.cfg.order in [1]:
            raise Exception("Unsupported ibe order {} (should be 1)".format(self.cfg.order))

    @staticmethod
    def _provides(cfg):
        if cfg.order:
            return (), ('ibd_xsec', 'ee', 'enu', 'jacobian', 'enuToEvis0')
        else:
            return (), ('ibd_xsec', 'ee', 'enu')

    def load_external_xsec(self):
        with TFileContext(self.cfg.filename) as f:
            if self.cfg.get('verbose', False):
                print('Load IBD xsec {} from {}'.format(self.cfg.objectname, self.cfg.filename))
            self._xsec_ext_tf = f.Get(self.cfg.objectname).Clone()
            assert self._xsec_ext_tf

        self.xsec_ext = C.TransformationTF2(self._xsec_ext_tf)

    def build(self):
        self.load_external_xsec()

        with entryContext(subgraph="IBD"):
            # initalize Evis to Enu converter
            # with self.namespace("ibd"):
            self.econv = R.EvisToEe()

            # register it's input and output
            self.set_input('ee', None, self.econv.Ee.Evis, argument_number=0)
            self.set_output('ee', None, self.econv.Ee.Ee)

            # if self.cfg.order==0:
                # # in 0th order and 1d case
                # # create 1d cross section
                # # with self.namespace("ibd"):
                # self.ibd = R.IbdZeroOrder()

                # # register Enu input and output
                # self.set_input('enu', None, self.ibd.Enu.Ee, argument_number=0)
                # self.set_output('enu', None, self.ibd.Enu.Enu)

                # # register cross section input and output
                # self.set_input('ibd_xsec', None, self.ibd.xsec.Ee, argument_number=0)
                # self.set_output('ibd_xsec', None, self.ibd.xsec.xsec)

                # #label cross section for the graph
                # self.ibd.xsec.setLabel('IBD xsec (0)')
            # elif self.cfg.order==1:
            # in 1th order and 2d case
            # create 2d cross section
            # with self.namespace("ibd"):
            self.ibd = R.IbdFirstOrder()

            # register Enu inputs and output
            self.set_input('enu', None, self.ibd.Enu.Ee, argument_number=0)
            self.set_input('enu', None, self.ibd.Enu.ctheta, argument_number=1)
            self.set_output('enu', None, self.ibd.Enu.Enu)

            # register cross section inputs and output
            # self.set_input('ibd_xsec', None, self.ibd.xsec.Enu, argument_number=0)
            # self.set_input('ibd_xsec', None, self.ibd.xsec.ctheta, argument_number=1)
            # self.set_output('ibd_xsec', None, self.ibd.xsec.xsec)
            self.set_input('ibd_xsec', None, self.xsec_ext.tf2.arg0, argument_number=0)
            self.set_input('ibd_xsec', None, self.xsec_ext.tf2.arg1, argument_number=1)
            self.set_output('ibd_xsec', None, self.xsec_ext.tf2.result)
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

    def define_variables(self):
        pass
