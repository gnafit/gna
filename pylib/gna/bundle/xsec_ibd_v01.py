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

            self.set_input(self.ibd.xsec.Ee,   'ibd_xsec', clone=0)
            self.set_output(self.ibd.xsec.xsec, 'ibd_xsec')

            self.ibd.xsec.setLabel('IBD xsec (0)')

        elif self.cfg.order==1:
            with self.ns("ibd"):
                self.ibd = ROOT.IbdFirstOrder()

            self.ibd.Enu.Ee(econv.Ee.Ee)
            self.ibd.xsec.Enu(self.ibd.Enu)
            # ibd.xsec.ctheta(integrator.points.y)

            # # ibd.jacobian.Enu(ibd.Enu)
            # # ibd.jacobian.Ee(integrator.points.x)
            # # ibd.jacobian.ctheta(integrator.points.y)
            # #
            # ibd.jacobian.setLabel('Jacobian')

        #
        # self.comp0 = R.FillLike(1.0)
        # self.comp0.fill.setLabel('OP comp0')

        # for i, it in enumerate(self.idx.iterate()):
            # dist_it = it.get_sub( ('d', 'r') )
            # dist = dist_it.current_format('baseline{autoindex}')

            # oscprobkey = dist_it.current_format('{autoindex}')[1:]
            # oscprob = self.objects.get( oscprobkey, None )
            # if not oscprob:
                # # with self.common_namespace(distenv):
                # oscprob = self.objects[oscprobkey] = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), dist)

            # component = it.get_current('c')
            # if component=='comp0':
                # output = self.comp0.fill.outputs['a']
                # input = self.comp0.fill.inputs['a']
            # else:
                # if not component in oscprob.transformations:
                    # raise Exception( 'No component %s in oscprob transformation'%component )

                # trans = oscprob.transformations[component]
                # trans.setLabel( it.current_format('OP {component}:\n{reactor}->{detector}') )
                # output = trans[component]
                # input  = trans['Enu']

            # if self.context:
                # self.context.set_output(output, self.cfg.name, it)
                # self.context.set_input(input, self.cfg.name, it, clone=0)

    def define_variables(self):
        from gna.parameters import ibd
        ibd.reqparameters(self.common_namespace('ibd'))

