# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class xsec_ibd_v01(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

    def build(self):
        self.econv = ROOT.EvisToEe()

        if ibdtype == 'zero':
            with self.ns("ibd"):
                ibd = ROOT.IbdZeroOrder()

            # econv.Ee.Evis(integrator.points.x)

            # ibd.Enu.Ee(econv.Ee.Ee)
            # ibd.xsec.Ee(econv.Ee.Ee)
        elif ibdtype == 'first':
            with self.ns("ibd"):
                ibd = ROOT.IbdFirstOrder()

            # econv.Ee.Evis(integrator.points.x)
            # ibd.Enu.Ee(econv.Ee.Ee)
            # ibd.Enu.ctheta(integrator.points.y)

            ibd.xsec.Enu(ibd.Enu)
            ibd.xsec.ctheta(integrator.points.y)

            # ibd.jacobian.Enu(ibd.Enu)
            # ibd.jacobian.Ee(integrator.points.x)
            # ibd.jacobian.ctheta(integrator.points.y)
            #
            ibd.jacobian.setLabel('Jacobian')
        else:
            raise Exception("unknown ibd type {0!r}".format(ibdtype))

        ibd.xsec.setLabel('IBD xsec')
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
        pass

