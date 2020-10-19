# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as N
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna.env import env, namespace
from gna.configurator import NestedDict
from collections import OrderedDict
from gna.bundle import TransformationBundle
from collections import Iterable, Mapping

class energy_nonlinearity_db_root_v04(TransformationBundle):
    """Detector energy nonlinearity parametrized via few curves (Daya Bay approach)

    The bundle uses HistNonlinearityB to make the conversion. It also uses to instances of
    InterpLinear to prepare the inputs.

    Changes since v03:
    - Switched to HistNonlinearityB
    - Does not support other major indices, apart from component
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

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
            return ('escale', 'lsnl_weight'), ('lsnl', 'lsnl_x', 'lsnl_component_y', 'lsnl_interpolator', 'lsnl_edges')
        else:
            return ('lsnl_weight',), ('lsnl', 'lsnl_component', 'lsnl_edges')

    def build(self):
        self.objects=NestedDict()
        data = self.load_data()

        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        for i, itl in enumerate(self.component_idx.iterate()):
            name, = itl.current_values()
            if not name in data:
                raise self._exception('The nonlinearity curve {} is not provided'.format(name))

            try:
                x, y = data[name]
            except:
                raise Exception('Unable to get x,y for nonlinearity {}'.format(name))

            Y = C.Points(y*x)

            if i:
                label=itl.current_format('NL correction {autoindex}')
            else:
                label=itl.current_format('NL nominal ({autoindex})')

                X = C.Points(x)
                X.points.setLabel(label+' X')
                self.set_output('lsnl_x', None, X.single())
                self.objects[('curves', name, 'X')] = X

            Y.points.setLabel(label+' Y')
            self.set_output('lsnl_component_y', itl, Y.single())
            self.objects[('curves', name, 'Y')] = Y

        #
        # Create direct and inverse interpolators
        #
        interp_direct  = C.InterpLinear(labels=('NL InSeg direct', 'NL interp direct'))
        interp_inverse = C.InterpLinear(labels=('NL InSeg inverse', 'NL interp inverse'))
        self.objects['interp_direct']=interp_direct
        self.objects['interp_inverse']=interp_inverse

        #
        # Interp_interpolator(xcoarse, ycoarse, newx)
        # x, y -> interp_direct  -> interp_direct(bin edges)
        # y, x -> interp_inverse -> interp_direct(bin edges)
        self.set_input('lsnl_interpolator', None, (interp_direct.insegment.edges, interp_direct.interp.x,     interp_inverse.interp.y),                                    argument_number=0)
        self.set_input('lsnl_interpolator', None, (interp_direct.interp.y,                                    interp_inverse.insegment.edges, interp_inverse.interp.x),     argument_number=1)
        self.set_input('lsnl_interpolator', None, (interp_direct.insegment.points, interp_direct.interp.newx, interp_inverse.insegment.points, interp_inverse.interp.newx), argument_number=2)

        self.set_output('lsnl_direct', None, interp_direct.interp.interp)
        self.set_output('lsnl_inverse', None, interp_inverse.interp.interp)

        expose_matrix = self.cfg.get('expose_matrix', False)
        with self.namespace:
            for i, itd in enumerate(self.detector_idx.iterate()):
                """Finally, original bin edges multiplied by the correction factor"""
                """Construct the nonlinearity calss"""
                nonlin = R.HistNonlinearityB(expose_matrix, labels=itd.current_format('NL matrix {autoindex}'))
                try:
                    nonlin.set_range(*self.cfg.nonlin_range)
                except KeyError:
                    pass

                self.objects[('nonlinearity',)+itd.current_values()] = nonlin

                self.set_input('lsnl_edges', itd, nonlin.matrix.Edges,              argument_number=0)
                interp_direct.interp.interp  >> nonlin.matrix.EdgesModified
                interp_inverse.interp.interp >> nonlin.matrix.BackwardProjection
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

    def get_buffers_auto(self, (k, obj)):
        return k, get_buffers_graph_or_hist1(obj)

    def load_data(self):
        tfile = R.TFile( self.cfg.filename, 'READ' )
        if tfile.IsZombie():
            raise IOError( 'Can not read ROOT file: '+self.cfg.filename )

        if isinstance(self.cfg.names, (Mapping, NestedDict)):
            graphs = OrderedDict([(k, tfile.Get(v)) for k, v in self.cfg.names.items()])
        elif isinstance(self.cfg.names, Iterable):
            graphs = OrderedDict([(k, tfile.Get(k)) for k in self.cfg.names])
        else:
            raise self._exception('Invalid cfg.names option: not mapping and not iterable')

        if not all( graphs.values() ):
            raise IOError( 'Some objects were not read from file: '+self.cfg.filename )

        graphs = OrderedDict(map(self.get_buffers_auto, graphs.items()))
        self.check_same_x(graphs)
        self.make_diff(graphs)

        tfile.Close()
        return graphs

    def make_diff(self, graphs):
        names = self.cfg.names
        nom, others = names[0], names[1:]

        nominal = graphs[nom][1]

        for name in others:
            y = graphs[name][1]
            y-=nominal

    def check_same_x(self, graphs):
        xcommon = None
        for name, (x, y) in graphs.items():
            if xcommon is None:
                xcommon = x
                continue

            if not N.allclose(xcommon, x, rtol=0, atol=1.e-16):
                raise self.exception('Nonlinearity curves X should be the same')

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

    def interpolate(self, (x, y), edges):
        fill_ = self.cfg.get('extrapolation_strategy', 'extrapolate')
        fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value=fill_ )
        res = fcn( edges )
        return res
