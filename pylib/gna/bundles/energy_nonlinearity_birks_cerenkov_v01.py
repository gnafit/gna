# reimplementation of ../bundles_legacy/detector_nonlinearity_db_root_v02

# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as N
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph
from gna.env import env, namespace
from gna.configurator import NestedDict
from collections import OrderedDict
from gna.bundle import TransformationBundle

class energy_nonlinearity_db_root_v02(TransformationBundle):
    debug = False
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.storage=NestedDict()

    @staticmethod
    def _provides(cfg):
        return (), ()

    def build(self):
        # with self.namespace:
            # for i, itd in enumerate(self.detector_idx.iterate()):
                # """Finally, original bin edges multiplied by the correction factor"""
                # """Construct the nonlinearity calss"""
                # nonlin = R.HistNonlinearity(self.debug, labels=itd.current_format('NL matrix\n{autoindex}'))
                # self.context.objects[('nonlinearity',)+itd.current_values()] = nonlin

                # self.set_input('lsnl_edges', itd, nonlin.matrix.Edges,         argument_number=0)
                # self.set_input('lsnl_edges', itd, nonlin.matrix.EdgesModified, argument_number=1)

                # trans = nonlin.smear
                # for j, itother in enumerate(self.nidx_minor.iterate()):
                    # it = itd+itother
                    # if j:
                        # trans = nonlin.add_transformation()
                        # nonlin.add_input()
                    # trans.setLabel(it.current_format('NL\n{autoindex}'))

                    # self.set_input('lsnl', it, trans.Ntrue, argument_number=0)
                    # self.set_output('lsnl', it, trans.Nrec)

    def define_variables(self):
        pass
