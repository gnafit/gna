# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as np
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna.env import env, namespace
from gna.configurator import NestedDict
from collections import OrderedDict
from gna.bundle import TransformationBundle
from collections import Iterable, Mapping

class energy_nonlinearity_db_root_subst_v01(TransformationBundle):
    """Detector energy nonlinearity parametrized via few curves (Daya Bay approach)

    The bundle provides the inverse energy scale converion Evisible(Eqed)
    defined by inverting and interpolating the Eqing(Evisible) curve.
    Evisible is the total positron energy + electron mass: Evis=Ee+me.

    This plugin also keeps the HistNonlinearityB instance as well.

    Changes since energy_nonlinearity_db_root_v04:
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
        transs = ('lsnl', 'lsnl_x', 'lsnl_component_y', 'lsnl_component_grad',
                  'lsnl_interpolator',
                  'lsnl_interpolator_grad',
                  'lsnl_edges')
        pars   = ('lsnl_weight',)
        if 'par' in cfg:
            return ('escale',)+pars, transs

        return pars, transs

    def build(self):
        self.objects=NestedDict()
        data, gradients = self.load_data()

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

            Y = C.Points(y)
            Grad = C.Points(gradients[name])

            if i:
                label=itl.current_format('NL correction {autoindex}')
            else:
                label=itl.current_format('NL nominal ({autoindex})')

                X = C.Points(x)
                X.points.setLabel(label+' X')
                self.set_output('lsnl_x', None, X.single())
                self.objects[('curves', name, 'X')] = X

            Y.points.setLabel(label+' Y')
            Grad.points.setLabel(label+' gradient')
            self.set_output('lsnl_component_y', itl, Y.single())
            self.set_output('lsnl_component_grad', itl, Grad.single())
            self.objects[('curves', name, 'Y')] = Y
            self.objects[('curves', name, 'Grad')] = Grad

        #
        # Create direct and inverse interpolators
        #
        interp_direct  = C.InterpLinear(labels=('NL InSeg direct', 'NL interp direct'))
        interp_inverse = C.InterpLinear(labels=('NL InSeg inverse', 'NL interp inverse'))
        self.objects['interp_direct']=interp_direct
        self.objects['interp_inverse']=interp_inverse

        interp_evis = C.InterpLinear(labels=('NL InSeg Evis(Eq)', 'Evis(Eq)'))
        self.objects['interp_evis']=interp_evis

        interp_deq_devis = C.InterpLinear(labels=('NL InSeg dEq/dEvis)', 'dEq/dEvis'))
        self.objects['interp_deq_devis']=interp_deq_devis

        # Interp_interpolators
        # interp_evis acts similarly to interp_invers, but expects different interpolation points
        #
        # x, y     -> interp_direct  : interp_direct(bin edges)
        # y, x     -> interp_inverse : interp_direct(bin edges)
        # y, x     -> interp_evis    : interp_evis(integration points of Eq)
        # x, dy/dx -> interp_grad    : interp_grad(Evis(integration points of Eq))
        #
        # Arguments: Evis, Eq, E edges, E integration points
        # Distribution:
        #   Interpolator     Xorig   Yorig      Xnew
        #   interp_direct    Evis    Eq         E edges
        #   interp_inverse   Eq      Evis       E edges
        #   interp_evis      Eq      Evis       E integration points
        #   interp_grad      Evis    dEq/dEvis  interp_evis: Evis(Eq=E integration points)
        #
        arg0=     (interp_direct.insegment.edges, interp_direct.interp.x)
        arg0=arg0+(interp_inverse.interp.y,)
        arg0=arg0+(interp_evis.interp.y,)
        arg0=arg0+(interp_deq_devis.insegment.edges, interp_deq_devis.interp.x)
        arg1=     (interp_direct.interp.y,)
        arg1=arg1+(interp_inverse.insegment.edges, interp_inverse.interp.x)
        arg1=arg1+(interp_evis.insegment.edges, interp_evis.interp.x)
        arg2=     (interp_direct.insegment.points, interp_direct.interp.newx)
        arg2=arg2+(interp_inverse.insegment.points, interp_inverse.interp.newx)
        arg3=     (interp_evis.insegment.points, interp_evis.interp.newx)
        self.set_input('lsnl_interpolator', None, arg0, argument_number=0)
        self.set_input('lsnl_interpolator', None, arg1, argument_number=1)
        self.set_input('lsnl_interpolator', None, arg2, argument_number=2)
        self.set_input('lsnl_interpolator', None, arg3, argument_number=3)

        arg0=interp_deq_devis.interp.y
        self.set_input('lsnl_interpolator_grad', None, arg0, argument_number=0)

        self.set_output('lsnl_direct',  None, interp_direct.interp.interp)
        self.set_output('lsnl_inverse', None, interp_inverse.interp.interp)
        self.set_output('lsnl_evis',    None, interp_evis.interp.interp)
        self.set_output('lsnl_interpolator_grad', None, interp_deq_devis.interp.interp)

        interp_evis.interp.interp >> (interp_deq_devis.insegment.points, interp_deq_devis.interp.newx)

        expose_matrix = self.cfg.get('expose_matrix', False)
        for i, itd in enumerate(self.detector_idx.iterate()):
            """Finally, original bin edges multiplied by the correction factor"""
            """Construct the nonlinearity calss"""
            with self.namespace:
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

        def mult((x, y)): y*=x
        def grad((k, (x,y))): return k, np.gradient(y, x[1]-x[0])

        graphs = OrderedDict(map(self.get_buffers_auto, graphs.items()))
        self.check_same_x(graphs)
        self.make_diff(graphs)
        graphs = self.supersample(graphs)
        map(mult, graphs.values())
        gradients = OrderedDict(map(grad, graphs.items()))

        tfile.Close()
        return graphs, gradients

    def make_diff(self, graphs):
        names = self.cfg.names
        if isinstance(names, (dict,NestedDict)):
            names = list(names.keys())
        nom, others = names[0], names[1:]

        nominal = graphs[nom][1]

        for name in others:
            y = graphs[name][1]
            y-=nominal

    def supersample(self, graphs):
        newgraphs = OrderedDict()

        x = graphs.values()[0][1]
        return graphs

    def check_same_x(self, graphs):
        xcommon = None
        for name, (x, y) in graphs.items():
            if xcommon is None:
                xcommon = x
                continue

            if not np.allclose(xcommon, x, rtol=0, atol=1.e-16):
                raise self.exception('Nonlinearity curves X should be the same')

        widths = xcommon[1:]-xcommon[:-1]
        assert np.allclose(widths, widths[0], rtol=0, atol=1.e-15), "LSNL steps should be similar"

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
