# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram
from gna.configurator import NestedDict

from gna.bundle import *
from gna.bundle.connections import pairwise

class root_histograms_v01(TransformationBundle):
    def __init__(self, **kwargs):
        variants = kwargs['cfg'].get('variants', None)
        if variants:
            namespaces = kwargs.pop( 'namespaces', None )
            if namespaces:
                raise Exception('root_histograms_v01 initializes namespaces on its own')
            kwargs['namespaces']=variants
        super(root_histograms_v01, self).__init__( **kwargs )

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print( 'Read input file {}:'.format(file.GetName()) )

        variants = self.cfg.get('variants', [self.common_namespace.name])
        for var in variants:
            fmt = var
            if isinstance(variants, (dict, NestedDict)):
                fmt = variants[var]

            hname = self.cfg.format.format(fmt)
            h = file.Get( hname )
            if not h:
                raise Exception('Can not read {hist} from {file}'.format( hist=hname, file=file.GetName() ))

            print( '  read{}: {}'.format(var and ' '+var or '', hname) )
            edges = get_bin_edges_axis( h.GetXaxis() )
            data  = get_buffer_hist1( h )
            if self.cfg.get( 'normalize', False ):
                data=data/data.sum()

            hist=Histogram(edges, data, ns=self.common_namespace(var))

            self.objects[('hist',var)]    = hist
            self.transformations_out[var] = hist.hist
            self.outputs[var]             = hist.hist.hist

        file.Close()
