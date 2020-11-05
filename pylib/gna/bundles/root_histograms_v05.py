from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.constructors import Histogram
from gna.configurator import NestedDict
from gna.grouping import Categories
import itertools as I

from gna.bundle import TransformationBundle

class root_histograms_v05(TransformationBundle):
    """Load ROOT histograms from ROOT files v05

    Updates:
        v04 - scale X/Y axes with 'xscale' and 'yscale' options
        v05 - Read a bunch of histograms
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.groups = Categories(self.cfg.get('groups', {}), recursive=True)

    @staticmethod
    def _provides(cfg):
        return (), tuple(cfg.names)

    def build(self):
        file = R.TFile( self.cfg.filename, 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print( 'Read input file {}:'.format(file.GetName()) )

        normalize=self.cfg.get('normalize', False)
        xscale = self.cfg.get('xscale', None)
        yscale = self.cfg.get('yscale', None)
        for name, format, labelfmt in I.izip_longest(self.cfg.names, self.cfg.formats, self.cfg.get('labels', [])):
            for it in self.nidx.iterate():
                if it.ndim()>0:
                    subst, = it.current_values()
                else:
                    subst = ''
                hname = self.groups.format(subst, format)
                h = file.Get(hname)
                if not h:
                    raise Exception('Can not read {hist} from {file}'.format(hist=hname, file=file.GetName()))

                print( '  read{}: {}'.format(' '+subst if subst else '', hname), end=' ' )

                edges = get_bin_edges_axis( h.GetXaxis() )
                data  = get_buffer_hist1( h )
                if normalize:
                    print( '[normalized]', end=' ' )
                    data=N.ascontiguousarray(data, dtype='d')
                    data=data/data.sum()
                else:
                    print()

                if xscale is not None:
                    print( '[xscale]', end=' ' )
                    edges*=xscale
                if yscale is not None:
                    print( '[yscale]', end=' ' )
                    data*=yscale

                print()

                if labelfmt is None:
                    labelfmt = 'hist {name}\n{autoindex}'
                hist=Histogram(edges, data, labels=it.current_format(labelfmt, name=name))
                self.set_output(name, it, hist.single())

                self.context.objects[('hist',subst)]    = hist

        file.Close()
