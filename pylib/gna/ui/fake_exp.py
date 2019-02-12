# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.ui import basecmd

import numpy as np
import ROOT
import gna.constructors as C
import mpl_tools.root2numpy as r2n
from gna.env import env
r2n.bind()

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-f', '--file', action='append', default=[],
                            dest='sources', help="Path to file with data to serve as fake observables")
        parser.add_argument('--ns', default='fake_data', type=env.ns, help="Name of the namespace in which fake data will be loaded")
        parser.add_argument('--filter', dest='filters', action='append', default=[],
                           help="Filters to select specific entries from ROOT files")
        parser.add_argument('--match-all', action="store_true", help="If set "
                            "combine filters as logical AND, otherwise logical OR")

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    def init(self):
        for source in self.opts.sources:
            if source.endswith(".root"):
                self._handler_root(source)
            elif source.endswith(".dat"):
                self._handler_dat(source)
            else:
                raise Exception("Unrecognized data type")


    def _handler_root(self, source):
        def _apply_filters(names):
            strategy = {False: any, True: all}
            for name in names:
                if strategy[self.opts.match_all](filt in name for filt in self.opts.filters):
                    yield name 
                #  for filt in self.opts.filters: 
                    #  if filt in name:

        roo_file = ROOT.TFile(source)
        items_in_file = [_.GetName() for _ in iter(roo_file.GetListOfKeys())]

        filtered = list(_apply_filters(items_in_file))
        print(filtered)
        # register observables
        for name in filtered:
            hist = roo_file.Get(name)
            data, edges = hist.get_buffer(), hist.get_edges()
            obs = C.Histogram(edges, data)
            self.opts.ns.addobservable(name, obs.hist)
            #  histed = C.Histogram()
        #  import IPython
        #  IPython.embed()

        
        

   

