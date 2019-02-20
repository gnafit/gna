# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.ui import basecmd

import numpy as np
import ROOT
import gna.constructors as C
from gna.exp import baseexp
import mpl_tools.root2numpy as r2n
r2n.bind()
from gna.env import env
import fnmatch as fn
from itertools import chain

class exp(baseexp):
    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('datapath', nargs='+', dest='sources', help="Path to file with data to serve as fake observables")
        parser.add_argument('--ns', default='fake_data', type=env.ns, help="Name of the namespace in which fake data will be loaded")
        parser.add_argument('--filter', dest='filters', action='append', default=[],
                           help="Filters to select specific entries from ROOT files")

    def __init__(self, namespace, opts):
        baseexp.__init__(self, namespace, opts)
        for source in self.opts.sources:
            if source.endswith(".root"):
                self._handler_root(source)
            elif source.endswith(".dat"):
                self._handler_dat(source)
            else:
                raise Exception("Unrecognized data type")


    def _handler_dat(self, source):
        raise NotImplementedError("Still not implemented")

    def _handler_root(self, source):
        def _apply_filters(names):
            for filt in self.opts.filters:
                    yield fn.filter(names, filt)

        roo_file = ROOT.TFile(source)
        items_in_file = [_.GetName() for _ in iter(roo_file.GetListOfKeys())]

        filtered = list(chain(*_apply_filters(items_in_file)))

        # register observables
        for name in filtered:
            hist = roo_file.Get(name)
            data, edges = hist.get_buffer(), hist.get_edges()
            obs = C.Histogram(edges, data)
            self.opts.ns.addobservable(name, obs.hist)

        
        

   

