# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram

from gna.bundle import *
from gna.bundle.connections import pairwise

class root_histograms_v01(TransformationBundle):
    name = 'root_histograms'
    def __init__(self, **kwargs):
        super(root_histograms_v01, self).__init__( **kwargs )

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print( 'Read input file {}:'.format(file.GetName()) )

        self.transformations=OrderedDict()
        for var in self.cfg.variants:
            hname = self.cfg.format.format(var)
            h = file.Get( hname )
            if not h:
                raise Exception('Can not read {hist} from {file}'.format( hist=hname, file=file.GetName() ))

            print( '  read', var, ':', hname )
            edges = get_bin_edges_axis( h.GetXaxis() )
            data  = get_buffer_hist1( h )
            hist = Histogram( edges, data )

            self.transformations[var] = hist
            self.output_transformations+=hist,

        for i, ns in enumerate(self.iterate_namespaces()):
            self.inputs += None,
            self.outputs += self.transformations[ns.name].hist,

        file.Close()
