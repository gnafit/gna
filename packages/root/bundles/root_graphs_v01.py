from load import ROOT as R
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna import constructors as C
from gna.grouping import Categories
import itertools as I
from gna.bundle import TransformationBundle
import numpy as np

from gna.configurator import StripNestedDict, NestedDict
from tools.schema import Schema, Optional, And, Or, Use
from tools.schema import isrootfile, isreadable, haslength
from typing import Tuple, Callable, Mapping, Union, Type

from ROOT import TGraph, TFile, TH1

StrOrPairOfStrings = Or(And(str, Use(lambda s: (s+'_x',s+'_y'))), And((str,), haslength(exactly=2)))

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

class root_graphs_v01(TransformationBundle):
    """Load ROOT graphs from a ROOT files v01

    Based on root_histograms_v05
    """
    def __init__(self, *args, **kwargs) -> None:
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.vcfg: dict = self._validator.validate(StripNestedDict(self.cfg))

        self.groups = Categories(self.vcfg['groups'], recursive=True)

    _validator = Schema(And({
                'bundle': object,
                'filename': And(isrootfile, isreadable),              # Input filename .root
                'names': [StrOrPairOfStrings],                        # list of names or pairs of names for x and y (_x/_y will be appended to single strings)
                'formats': [str],                                     # list of formats, defining keys to read from a file
                Optional('labels', default=[]): [StrOrPairOfStrings], # list of labels (format) or pairs of labels (formats) (_x/_y will be appended to single strings)
                Optional('debug_show', default=False): bool,          # debug_show plots with matplotlib
                Optional('groups', default={}): {},                   # Overriding is disabled
                Optional('preinterpolate', default=False): Or({       # Pre interpolate data
                    'npoints': int,                                   # Number of points in each input interval (incliding the bounding points)
                    'interp_kwargs': {                                # Interpolator kwargs (scipy.interpolate)
                        'kind': str,                                  # Interpolator kind: scipy.interp1d options + 'akima' or 'pchip'
                        Optional(str): object                         # All other interpolator options
                        }
                    },
                    False)
            },
            fields_have_same_length('names', 'formats'),
            fields_have_same_length('names', 'labels', permit_empty=True)
            )
        )

    @classmethod
    def _provides(cls, cfg: NestedDict) -> Tuple[Tuple, Tuple]:
        cfg: dict = cls._validator.validate(StripNestedDict(cfg))
        return (), tuple(I.chain(*cfg['names']))

    def build(self) -> None:
        file = TFile( self.vcfg['filename'], 'READ' )
        if file.IsZombie():
            raise Exception('Can not read ROOT file '+file.GetName())

        print('Read input file {}:'.format(file.GetName()))

        names = self.vcfg['names']
        formats = self.vcfg['formats']
        labels = self.vcfg['labels']
        for name, format, labelfmt in zip(names, formats, labels):
            for it in self.nidx.iterate():
                if it.ndim()>0:
                    subst, = it.current_values()
                else:
                    subst = ''
                gname = self.groups.format(subst, format)
                graph = file.Get(gname)
                if not graph:
                    raise Exception('Can not read {hist} from {file}'.format(hist=gname, file=file.GetName()))

                print( '  read{}: {}'.format(' '+subst if subst else '', gname), end=' ' )

                pointsx, pointsy = self.get_data(graph, name, it, labelfmt)
                print()

                self.set_output(name[0], it, pointsx.single())
                self.set_output(name[1], it, pointsy.single())

        file.Close()

        if self.vcfg['debug_show']:
            from matplotlib import pyplot as plt
            plt.show()


    def get_data(self, graph: Union[TGraph, TH1], name: Tuple[str, str], it: Type, labelsfmt: Union[Tuple[str,str], None]) -> Tuple[Type, Type]:
        x0, y0 = get_buffers_graph_or_hist1(graph)

        #
        # Interpolate
        #
        x, y = self.interpolate(x0, y0)

        #
        # Make labels
        #
        if labelsfmt is None:
            labelfmtx = 'X: {name}\n{autoindex}'
            labelfmty = 'Y: {name}\n{autoindex}'
        else:
            labelfmtx, labelfmty = labelsfmt

        labelx = it.current_format(labelfmtx, name=name[0])
        labely = it.current_format(labelfmty, name=name[1])

        #
        # Plot debug data
        #
        if self.vcfg['debug_show']:
            from matplotlib import pyplot as plt
            plt.figure()
            ax = plt.subplot(111, xlabel=labelx, ylabel=labely, title=f"{name[1]} vs. {name[0]}")
            ax.grid()
            m=ax.plot(x0, y0, 'o', markerfacecolor='none', label='input')
            color=m[0].get_color()
            ax.plot(x, y, '-', markerfacecolor='none', label='interpolation')

            ax.legend()

        #
        # Results
        #
        Px = C.Points(x, labels=labelx)
        Py = C.Points(y, labels=labely)
        # hist=Points(edges, data, labels=it.current_format(labelfmt, name=name))
        # self.context.objects[('hist',subst)]    = hist

        return Px, Py

    def interpolate(self, x0: np.ndarray, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        interp_opts = self.vcfg['preinterpolate']
        if not interp_opts:
            return x0, y0

        npoints_sub = interp_opts['npoints']
        if npoints_sub<3:
            raise self.exception(f'npoints should be >2, got {npoints_sub}')

        nsegments_sub = npoints_sub-1
        nsegments_in = x0.size-1
        nsegments_out = nsegments_in*nsegments_sub
        newx = np.empty(nsegments_out+1, dtype='d')

        left, right = x0[:-1], x0[1:]
        newx_data = np.linspace(left, right, npoints_sub, axis=1)

        newx[:-1] = newx_data[:,:-1].flatten()
        newx[-1] = newx_data[-1,-1]

        from scipy import interpolate as si
        interp_kwargs = interp_opts['interp_kwargs']
        kind = interp_kwargs.pop('kind')
        if kind=='pchip':
            interpolator = si.PchipInterpolator(x0, y0)
        elif kind=='akima':
            interpolator = si.Akima1DInterpolator(x0, y0)
        else:
            interpolator = si.interp1d(x0, y0, kind=kind, **interp_kwargs)

        newy = interpolator(newx)

        return newx, newy


