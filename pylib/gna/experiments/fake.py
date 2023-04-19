from gna.ui import basecmd
import numpy as np
import ROOT
import gna.constructors as C
from gna.exp import baseexp
from gna.env import env
import fnmatch as fn
from itertools import chain
from tools.dictwrapper import DictWrapper

class exp(baseexp):
    _data: list

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('sources', nargs='+', help="Paths to file with data to serve as fake observables")
        parser.add_argument('--take', nargs='+', default=[],
                            help="Filters to select specific entries from ROOT files")
        parser.add_argument('-f', '--forward', nargs='+', default=[], help='Forward the data to a set of inputs from the future env', metavar='path')
        parser.add_argument('-r', '--root', default='spectra', help='root path')
        parser.add_argument('-v', '--verbose', default=0, action='count', help='verbosity')

    def _iterate_forward_objects(self):
        for path in self.opts.forward:
            place = env.future[(self.opts.root, path)]
            if isinstance(place, DictWrapper):
                yield from place.walkvalues()
            else:
                yield place

    def init(self):
        self._data = []

        if self.opts.forward:
            self._input_iterator = self._iterate_forward_objects()
        else:
            self._input_iterator = None

        for idx, source in enumerate(self.opts.sources):
            if source.endswith(".root"):
                self._handler_root(source)
            elif source.endswith(".dat"):
                self._handler_dat(source)
            elif source.endswith(".tsv"):
                self._handler_tsvhist(source, idx)
            else:
                raise Exception("Unrecognized data type")

        if self._input_iterator:
            try:
                inp = next(self._input_iterator)
            except StopIteration:
                pass
            else:
                raise Exception('Fake.forward: insufficient number of data items to fill all inputs')


    def add_observable(self, path, out):
        self.namespace.addobservable(path, out)

        env.future[(self.opts.root, self.namespace.path, path)]=out
        if self.opts.verbose:
            path='.'.join((self.opts.root, self.namespace.path, path))
            print(f'Add data as {path}: {out!s}')

        if self._input_iterator:
            try:
                inp = next(self._input_iterator)
            except StopIteration:
                raise Exception('Fake.forward: insufficient number of forward items')
            else:
                out >> inp

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

        obs = C.Histogram(edges, y, labels=source)
        self._data.append(obs)
        self.add_observable(name, obs.hist)

    def _handler_root(self, source):
        _, sourcename = source.rsplit('/', 1)

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
            obs = C.Histogram(hist, labels=f'{sourcename}:\\n{name}')
            self._data.append(obs)
            self.add_observable(name, obs.hist)

        roo_file.Close()
