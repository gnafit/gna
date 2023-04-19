from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.bundle import *
from gna.env import env
from tools.data_load import read_object_auto
from gna.configurator import StripNestedDict
from tools.schema import *

class load_graph_v01(TransformationBundle):
    """Load X,Y data

    Indexing:
      major - gets its own input file
      minor - forbidden
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'minor')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        self._load_data()

    _validator = Schema({
            'bundle': object,
            'name': str,
            'filenamefmt': isfilewithext('root'),
            'objectnamefmt': str,
            Optional('verbose', default=0): Or(int, And(bool, Use(int))),
            Optional('allowmissing', default=False): bool,
            Optional('labelx', default='{name}|{autoindex}'): str,
            Optional('labely', default='{name}|{autoindex}'): str,
            Optional('skip', default=[]): StrOrListOfStrings
        })

    @staticmethod
    def _provides(cfg):
        name = cfg['name']
        return (), (name+'_x', name+'_y')

    def _load_data(self):
        """Read raw input spectra"""
        filenamefmt = self.vcfg['filenamefmt']
        objnamefmt = self.vcfg['objectnamefmt']
        skiplist = self.vcfg['skip']
        label = ' '+self.vcfg['name']
        self._data = {}
        for it in self.nidx_major:
            filename = it.current_format(filenamefmt)
            objname  = it.current_format(objnamefmt)
            name = it.current_format()

            if name in skiplist:
                continue

            try:
                self._data[name] = \
                    read_object_auto(filename, name=objname, verbose=self.vcfg['verbose'], suffix=f' [{name}{label}]')
            except (IOError, ValueError):
                if self.vcfg['allowmissing']:
                    pass
                else:
                    raise

            assert len(self._data) != 0, "No data loaded"

    def build(self):
        name=self.vcfg['name']
        self.points = {}
        for it in self.nidx_major:
            dataname = it.current_format()

            try:
                x0, y0 = self._data[dataname]
            except KeyError:
                pass
            else:
                x = C.Points(x0, labels=it.current_format(self.vcfg['labelx'], name=name))
                y = C.Points(y0, labels=it.current_format(self.vcfg['labely'], name=name))

                self.points[name] = x,y

                self.set_output(f'{name}_x', it, x.single())
                self.set_output(f'{name}_y', it, y.single())

    def define_variables(self):
        pass

