from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.constructors import Histogram
from gna.grouping import Categories
from gna.bundle import TransformationBundle
import numpy as np

from gna.configurator import StripNestedDict, NestedDict
from tools.schema import Schema, Optional, And
from tools.schema import isrootfile, isreadable
from typing import Tuple, Callable, Mapping

from ROOT import TGraph, TFile
def fields_have_same_length(*fields: str, permit_empty: bool=False) -> Callable:
    def validator(d: Mapping) -> bool:
        first, rest = fields[0], fields[1:]
        n = len(d[first])

        if permit_empty:
            for key in rest:
                val = len(d[key])
                if val and val!=n:
                    return False

        return all(len(d[key])==n for key in rest)

    return validator

class root_histograms_v05(TransformationBundle):
    """Load ROOT histograms from ROOT files v05

    Updates:
        v04 - scale X/Y axes with 'xscale' and 'yscale' options
        v05 - Read a bunch of histograms, add validator
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.vcfg: dict = self._validator.validate(StripNestedDict(self.cfg))

        self.groups = Categories(self.vcfg['groups'], recursive=True)

    _validator = Schema(And({
                'bundle': object,
                'filename': And(isrootfile, isreadable),              # Input filename .root
                'names': [str],                                       # list of names
                'formats': [str],                                     # list of formats, defining keys to read from a file
                Optional('labels', default=[]): [str],                # list of labels (format)
                Optional('normalize', default=False): bool,           # normalize the histograms
                Optional('xscale', default=None): float,              # scale for X
                Optional('yscale', default=None): float,              # scale for Y
                Optional('groups', default={}): {},                   # Overriding is disabled
            },
            fields_have_same_length('names', 'formats'),
            fields_have_same_length('names', 'labels', permit_empty=True)
            )
        )

    @classmethod
    def _provides(cls, cfg: NestedDict) -> Tuple[Tuple, Tuple]:
        cfg: dict = cls._validator.validate(StripNestedDict(cfg))
        return (), tuple(cfg['names'])

    def build(self) -> None:
        file = TFile( self.vcfg['filename'], 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print('Read input file {}:'.format(file.GetName()))

        normalize = self.vcfg['normalize']
        xscale    = self.vcfg['xscale']
        yscale    = self.vcfg['yscale']

        names = self.vcfg['names']
        formats = self.vcfg['formats']
        labels = self.vcfg['labels']
        for name, format, labelfmt in zip(names, formats, labels):
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
                    data=np.ascontiguousarray(data, dtype='d')
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
