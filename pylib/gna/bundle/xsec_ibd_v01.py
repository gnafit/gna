# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class xsec_ibd_v01(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

        if not self.cfg.order in [0, 1]:
            raise Exception("Unsupported ibe order {} (should be 0 or 1)".format(self.cfg.order))

    def build(self):
        with self.common_namespace("ibd"):
            self.econv = R.EvisToEe()

        self.set_input( self.econv.Ee.Evis, 'ee', clone=0 )
        self.set_output( self.econv.Ee.Ee,  'ee' )

        if self.cfg.order==0:
            with self.common_namespace("ibd"):
                self.ibd = R.IbdZeroOrder()

            self.set_input(self.ibd.Enu.Ee,   'enu', clone=0)
            self.set_output(self.ibd.Enu.Enu, 'enu')

            self.set_input(self.ibd.xsec.Ee,    'ibd_xsec', clone=0)
            self.set_output(self.ibd.xsec.xsec, 'ibd_xsec')

            self.ibd.xsec.setLabel('IBD xsec (0)')
        elif self.cfg.order==1:
            with self.common_namespace("ibd"):
                self.ibd = R.IbdFirstOrder()

            self.set_input(self.ibd.Enu.Ee,     'enu', clone=0)
            self.set_input(self.ibd.Enu.ctheta, 'enu', clone=1)
            self.set_output(self.ibd.Enu.Enu,   'enu')

            self.set_input(self.ibd.xsec.Enu,    'ibd_xsec', clone=0)
            self.set_input(self.ibd.xsec.ctheta, 'ibd_xsec', clone=1)
            self.set_output(self.ibd.xsec.xsec,  'ibd_xsec')
            self.ibd.xsec.setLabel('IBD xsec (1)')

            self.set_input(self.ibd.jacobian.Enu,       'jacobian', clone=0)
            self.set_input(self.ibd.jacobian.Ee,        'jacobian', clone=1)
            self.set_input(self.ibd.jacobian.ctheta,    'jacobian', clone=2)
            self.set_output(self.ibd.jacobian.jacobian, 'jacobian')
            self.ibd.jacobian.setLabel('Ee->Enu jacobian')

        self.ibd.Enu.setLabel('Enu')

    def define_variables(self):
        from gna.parameters import ibd
        ibd.reqparameters(self.common_namespace('ibd'))

