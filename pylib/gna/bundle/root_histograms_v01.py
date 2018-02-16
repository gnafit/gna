# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram
from gna.configurator import NestedDict
from gna.grouping import Categories

from gna.bundle import *
from gna.bundle.connections import pairwise

class root_histograms_v01(TransformationBundle):
    def __init__(self, **kwargs):
        variants   = kwargs['cfg'].get('variants', None)
        namespaces = kwargs.get('namespaces', None)

        if variants and namespaces:
            print( 'root_histograms_v01 got namespaces and variants in the same time:' )
            print( '    variants:', variants )
            print( '    namespaces:', namespaces )
            raise Exception('root_histograms_v01 confusing initialization')

        if variants:
            kwargs['namespaces']=variants

        super(root_histograms_v01, self).__init__( **kwargs )

        self.groups = Categories( self.cfg.get('groups', {}), recursive=True )

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print( 'Read input file {}:'.format(file.GetName()) )

        variants = self.cfg.get('variants', self.namespaces)
        obsname = self.cfg.get('observable', '')
        for var in variants:
            if isinstance(var, basestring):
                ns = self.common_namespace(var)

                if isinstance(variants, (dict, NestedDict)):
                    subst = variants[var]
                else:
                    subst = var
            else:
                ns    = var
                var   = ns.name
                subst = var

            hname = self.groups.format(subst, self.cfg.format)
            # print( self.cfg.format, subst, hname )
            h = file.Get( hname )
            if not h:
                raise Exception('Can not read {hist} from {file}'.format( hist=hname, file=file.GetName() ))

            print( '  read{}: {}'.format(var and ' '+var or '', hname), end='' )
            edges = get_bin_edges_axis( h.GetXaxis() )
            data  = get_buffer_hist1( h )
            if self.cfg.get( 'normalize', False ):
                print( ' [normalized]' )
                data=N.ascontiguousarray(data, dtype='d')
                data=data/data.sum()
            else:
                print()

            hist=Histogram(edges, data, ns=ns)

            self.objects[('hist',var)]    = hist
            self.transformations_out[var] = hist.hist
            self.outputs[var]             = hist.hist.hist

            """Define observable"""
            if obsname:
                ns.addobservable( obsname, hist.hist.hist )

        file.Close()
