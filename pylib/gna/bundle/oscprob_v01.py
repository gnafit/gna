# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import constructors as C
from gna.bundle import *

class oscprob_v01(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

        self.idx = self.cfg.indices
        from gna.expression import NIndex
        if not isinstance(self.idx, NIndex):
            self.idx = NIndex(fromlist=self.cfg.indices)

    def build(self):
        self.comp0 = R.FillLike(1.0)
        self.comp0.fill.setLabel('OP comp0')

        for i, it in enumerate(self.idx.iterate()):
            dist_it = it.get_sub( ('d', 'r') )
            dist = dist_it.current_format('baseline{autoindex}')

            oscprobkey = dist_it.current_format('{autoindex}')[1:]
            oscprob = self.objects.get( oscprobkey, None )
            if not oscprob:
                # with self.common_namespace(distenv):
                oscprob = self.objects[oscprobkey] = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae(), dist)

            component = it.get_current('c')
            if component=='comp0':
                output = self.comp0.fill.outputs['a']
                input = self.comp0.fill.inputs['a']
            else:
                if not component in oscprob.transformations:
                    raise Exception( 'No component %s in oscprob transformation'%component )

                trans = oscprob.transformations[component]
                trans.setLabel( it.current_format('OP {component}:\n{reactor}->{detector}') )
                output = trans[component]
                input  = trans['Enu']

            if self.context:
                self.context.set_output(output, self.cfg.name, it)
                self.context.set_input(input, self.cfg.name, it, clone=0)

    def define_variables(self):
        from gna.parameters.oscillation import reqparameters
        reqparameters(self.common_namespace)

        for it in self.idx.get_sub( ['r', 'd'] ):
            key = it.current_format('baseline{autoindex}')

            self.common_namespace.reqparameter( key, central=1, sigma=0.1, fixed=True )

