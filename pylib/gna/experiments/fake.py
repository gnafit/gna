# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.ui import basecmd
import numpy as np
import ROOT
import gna.constructors as C
from gna.exp import baseexp
from gna.env import env
import fnmatch as fn
from itertools import chain

class exp(baseexp):
    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('sources', nargs='+', help="Paths to file with data to serve as fake observables")
        parser.add_argument('--take', nargs='+', default=[],
                            help="Filters to select specific entries from ROOT files")

    def init(self):
        self.data = []
        for idx, source in enumerate(self.opts.sources):
            if source.endswith(".root"):
                self._handler_root(source)
            elif source.endswith(".dat"):
                self._handler_dat(source)
            elif source.endswith(".tsv"):
                self._handler_tsvhist(source, idx)
            else:
                raise Exception("Unrecognized data type")

    def _handler_dat(self, source, idx):
        raise NotImplementedError("Still not implemented")

    def _handler_tsvhist(self, source, idx):
        data = np.loadtxt(source)
        x, y = data.T
        edges = np.concatenate( [x, [x[-1]+x[1]-x[0]]] )

        try:
            name = self.opts.take[idx]
        except KeyError:
            raise Exception('Name for input file %i is not provided'%idx)

        obs = C.Histogram(edges, y)
        self.data.append(obs)
        self.namespace.addobservable(name, obs.hist)

    def _handler_root(self, source):
        import mpl_tools.root2numpy as r2n
        r2n.bind()

        def _apply_filters(names):
            if not self.opts.take:
                yield names
            for filt in self.opts.take:
                    yield fn.filter(names, filt)

        roo_file = ROOT.TFile(source)
        items_in_file = [_.GetName() for _ in iter(roo_file.GetListOfKeys())]

        filtered = list(chain(*_apply_filters(items_in_file)))

        # register observables
        for name in filtered:
            hist = roo_file.Get(name)
            data, edges = hist.get_buffer(), hist.get_edges()
            obs = C.Histogram(edges, data)
            self.data.append(obs)
            self.namespace.addobservable(name, obs.hist)

        roo_file.Close()
