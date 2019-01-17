# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.constructors import Histogram
from gna.configurator import NestedDict
from gna.grouping import Categories

from gna.bundle import TransformationBundle

class root_histograms_v03(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.groups = Categories( self.cfg.get('groups', {}), recursive=True )

    @staticmethod
    def _provides(cfg):
        return (), (cfg.name,)

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print( 'Read input file {}:'.format(file.GetName()) )

        for it in self.nidx.iterate():
            if it.ndim()>0:
                subst, = it.current_values()
            else:
                subst = ''
            hname = self.groups.format(subst, self.cfg.format)
            h = file.Get( hname )
            if not h:
                raise Exception('Can not read {hist} from {file}'.format( hist=hname, file=file.GetName() ))

            print( '  read{}: {}'.format(' '+subst, hname), end='' )
            edges = get_bin_edges_axis( h.GetXaxis() )
            data  = get_buffer_hist1( h )
            if self.cfg.get( 'normalize', False ):
                print( ' [normalized]' )
                data=N.ascontiguousarray(data, dtype='d')
                data=data/data.sum()
            else:
                print()

            fmt = self.cfg.get('label', 'hist {name}\n{autoindex}')
            hist=Histogram(edges, data, labels=it.current_format(fmt, name=self.cfg.name))
            self.set_output(self.cfg.name, it, hist.single())

            self.context.objects[('hist',subst)]    = hist

        file.Close()
