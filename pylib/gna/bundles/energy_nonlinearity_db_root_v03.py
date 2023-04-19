from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as N
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna.env import env, namespace
from gna.configurator import NestedDict
from gna.bundle import TransformationBundle
from collections.abc import Iterable, Mapping

class energy_nonlinearity_db_root_v03(TransformationBundle):
    """Detector energy nonlinearity parametrized via few curves (Daya Bay approach)
    Based on: v02
    Changes:
    - support histograms in addition to TGraphs
    - add expose_matrix option (instead of debug)
    - par (escale) may be optional
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 2, 'major')

        if len(self.cfg.bundle.major)==2:
            detector_name, component_name = self.cfg.bundle.major
            self.detector_idx = self.nidx_major.get_subset(detector_name)
            self.component_idx = self.nidx_major.get_subset(component_name)
        elif len(self.cfg.bundle.major)==1:
            component_name = self.cfg.bundle.major
            self.detector_idx = self.nidx_major.get_subset([])
            self.component_idx = self.nidx_major.get_subset(component_name)
        else:
            raise self._exception('Unable to obtain major indices: detector and component')

        self.storage=NestedDict()

    @staticmethod
    def _provides(cfg):
        if 'par' in cfg:
            return ('escale', 'lsnl_weight'), ('lsnl', 'lsnl_component', 'lsnl_edges')
        else:
            return ('lsnl_weight',), ('lsnl', 'lsnl_component', 'lsnl_edges')

    def build_graphs( self, graphs ):
        #
        # Interpolate curves on the default binning
        # (extrapolate as well)
        #
        self.newx_out = self.context.outputs[self.cfg.edges]
        newx = self.newx_out.data()
        newy = dict()
        for name, xy in graphs.items():
            f = self.interpolate( xy, newx )
            newy[name]=f
            self.storage[name] = f.copy()

        #
        # All curves but first are the corrections to the nominal
        #
        newy_values = list(newy.values())
        for f in newy_values[1:]:
            f-=newy_values[0]

        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        for i, itl in enumerate(self.component_idx.iterate()):
            name, = itl.current_values()
            if not name in newy:
                raise self._exception('The nonlinearity curve {} is not provided'.format(name))
            y = newy[name]
            pts = C.Points( y, ns=self.namespace )
            if i:
                label=itl.current_format('NL correction {autoindex}')
            else:
                label=itl.current_format('NL nominal ({autoindex})')
            pts.points.setLabel(label)
            self.set_output('lsnl_component', itl, pts.single())
            self.context.objects[('curves', name)] = pts

        expose_matrix = R.GNA.DataPropagation.Propagate if self.cfg.get('expose_matrix', False) else R.GNA.DataPropagation.Ignore
        with self.namespace:
            for i, itd in enumerate(self.detector_idx.iterate()):
                """Finally, original bin edges multiplied by the correction factor"""
                """Construct the nonlinearity calss"""
                nonlin = R.HistNonlinearity(expose_matrix, labels=itd.current_format('NL matrix {autoindex}'))
                try:
                    nonlin.set_range(*self.cfg.nonlin_range)
                except KeyError:
                    pass

                self.context.objects[('nonlinearity',)+itd.current_values()] = nonlin

                self.set_input('lsnl_edges', itd, nonlin.matrix.Edges,         argument_number=0)
                self.set_input('lsnl_edges', itd, nonlin.matrix.EdgesModified, argument_number=1)
                self.set_output('lsnl_matrix', itd, nonlin.matrix.FakeMatrix)

                trans = nonlin.smear
                for j, itother in enumerate(self.nidx_minor.iterate()):
                    it = itd+itother
                    if j:
                        trans = nonlin.add_transformation()
                        nonlin.add_input()
                    trans.setLabel(it.current_format('NL {autoindex}'))

                    self.set_input('lsnl', it, trans.Ntrue, argument_number=0)
                    self.set_output('lsnl', it, trans.Nrec)

    def get_buffers_auto(self, kobj):
        k, obj = kobj
        return k, get_buffers_graph_or_hist1(obj)

    def build(self):
        tfile = R.TFile( self.cfg.filename, 'READ' )
        if tfile.IsZombie():
            raise IOError( 'Can not read ROOT file: '+self.cfg.filename )

        if isinstance(self.cfg.names, (Mapping, NestedDict)):
            graphs = dict([(k, tfile.Get(v)) for k, v in self.cfg.names.items()])
        elif isinstance(self.cfg.names, Iterable):
            graphs = dict([(k, tfile.Get(k)) for k in self.cfg.names])
        else:
            raise self._exception('Invalid cfg.names option: not mapping and not iterable')

        if not all( graphs.values() ):
            raise IOError( 'Some objects were not read from file: '+self.cfg.filename )

        graphs = dict(map(self.get_buffers_auto, graphs.items()))

        ret = self.build_graphs( graphs )

        tfile.Close()
        return ret

    def define_variables(self):
        par=None
        for itl in self.component_idx.iterate():
            if par is None:
                par = self.reqparameter('lsnl_weight', itl, central=1.0, fixed=True, label='Nominal nonlinearity curve weight ({autoindex})')
            else:
                par = self.reqparameter('lsnl_weight', itl, central=0.0, sigma=1.0, label='Correction nonlinearity weight for {autoindex}')


        if 'par' in self.cfg:
            if self.cfg.par.central!=1:
                raise self._exception('Relative energy scale parameter should have central value of 1 by definition')

            for it in self.detector_idx.iterate():
                self.reqparameter('escale', it, cfg=self.cfg.par, label='Uncorrelated energy scale for {autoindex}' )

    def interpolate(self, xy, edges):
        x, y = xy
        fill_ = self.cfg.get('extrapolation_strategy', 'extrapolate')
        fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value=fill_ )
        res = fcn( edges )
        return res
