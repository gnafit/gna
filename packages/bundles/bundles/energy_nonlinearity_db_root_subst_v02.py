from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as np
import gna.constructors as C
from gna.converters import convert
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna.env import env, namespace
from gna.bundle import TransformationBundle
from collections.abc import Iterable, Mapping
from tools.dictwrapper import DictWrapper

from gna.configurator import StripNestedDict
from tools.schema import *

class energy_nonlinearity_db_root_subst_v02(TransformationBundle):
    """Detector energy nonlinearity parametrized via few curves (Daya Bay approach)

    The bundle provides the inverse energy scale converion Evisible(Eqed)
    defined by inverting and interpolating the Eqing(Evisible) curve.
    Evisible is the total positron energy + electron mass: Evis=Ee+me.

    This plugin also keeps the HistNonlinearityB instance as well.

    Changes since energy_nonlinearity_db_root_subst_v01:
    - Add configuration validator
    - add an option to bypass energy nonlinearity for some values of the index
    - disable energy scale parameter
    - Minor:
        * define a configuration validator
        * replace NestedDict with DictWrapper
        * remove ordered dict
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

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        minor_extra = self.vcfg['minor_extra']
        if minor_extra:
            self.nidx_minor = self.nidx_minor + self.nidx.get_subset(minor_extra)
            if self.vcfg['verbose']:
                print('Extending minor indices with:', minor_extra)

    _validator = Schema({
            'bundle': object,
            'filename': And(isrootfile, isreadable),
            'names': Or({str: str}, And(Or([str], (str,)), Use(lambda l: {s:s for s in l}))),
            Optional('expose_matrix', default=False): bool,
            Optional('nonlin_range', default=None): And((float,), haslength(exactly=2)),
            Optional('supersample', default=None): int,
            # Optional('extrapolation_strategy', default='extrapolate'): str,
            Optional('minor_extra', default=()): Or([str], (str,), And(str, Use(lambda s: [s]))),
            Optional('bypass_minor', default=()): Or([str], (str,), And(str, Use(lambda s: [s]))),
            Optional('verbose', default=0): Or(int, And(bool, Use(int)))
        })

    @staticmethod
    def _provides(cfg):
        transs = ('lsnl', 'lsnl_x', 'lsnl_component_y', 'lsnl_component_grad',
                  'lsnl_interpolator',
                  'lsnl_interpolator_grad',
                  'lsnl_edges',
                  'lsnl_evis',
                  )
        pars   = ('lsnl_weight',)
        # if 'par' in cfg:
            # return ('escale',)+pars, transs

        return pars, transs

    def build(self):
        self.objects={}
        objects = DictWrapper(self.objects)
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
                objects[('curves', name, 'X')] = X

            Y.points.setLabel(label+' Y')
            Grad.points.setLabel(label+' gradient')
            self.set_output('lsnl_component_y', itl, Y.single())
            self.set_output('lsnl_component_grad', itl, Grad.single())
            objects[('curves', name, 'Y')] = Y
            objects[('curves', name, 'Grad')] = Grad

        #
        # Create direct and inverse interpolators
        #
        interp_direct  = C.InterpLinear(labels=('NL InSeg direct', 'NL interp direct'))
        interp_inverse = C.InterpLinear(labels=('NL InSeg inverse', 'NL interp inverse'))
        objects['interp_direct']=interp_direct
        objects['interp_inverse']=interp_inverse

        interp_evis = C.InterpLinear(labels=('NL InSeg Evis(Eq)', 'Evis(Eq)'))
        objects['interp_evis']=interp_evis

        interp_deq_devis = C.InterpLinear(labels=('NL InSeg dEq/dEvis)', 'dEq/dEvis'))
        objects['interp_deq_devis']=interp_deq_devis

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

        expose_matrix = R.GNA.DataPropagation.Propagate if self.vcfg['expose_matrix'] else R.GNA.DataPropagation.Ignore
        nonlin_range = self.vcfg['nonlin_range']
        bypass_minor = self.vcfg['bypass_minor']
        verbosity = self.vcfg['verbose']

        for i, itd in enumerate(self.detector_idx.iterate()):
            """Finally, original bin edges multiplied by the correction factor"""
            """Construct the nonlinearity calss"""
            with self.namespace:
                nonlin = R.HistNonlinearityB(expose_matrix, labels=itd.current_format('NL matrix {autoindex}'))
            if nonlin_range:
                nonlin.set_range(*nonlin_range)

            objects[('nonlinearity',)+itd.current_values()] = nonlin

            self.set_input('lsnl_edges', itd, nonlin.matrix.Edges,              argument_number=0)
            interp_direct.interp.interp  >> nonlin.matrix.EdgesModified
            interp_inverse.interp.interp >> nonlin.matrix.BackwardProjection
            self.set_output('lsnl_matrix', itd, nonlin.matrix.FakeMatrix)

            trans = nonlin.smear
            usedefaulttransformation = True
            for j, itother in enumerate(self.nidx_minor.iterate()):
                it = itd+itother

                minorstr = itother.current_format()

                if minorstr in bypass_minor:
                    if verbosity:
                        print(f'{self.__class__.__name__}: bypass {minorstr}')

                    view = C.View(labels='bypass LSNL')
                    objects[('view',)+it.current_values()]=view

                    self.set_input('lsnl', it, view.view.inputs.data, argument_number=0)
                    self.set_output('lsnl', it, view.view.view)

                else:
                    if usedefaulttransformation:
                        usedefaulttransformation = False
                    else:
                        trans = nonlin.add_transformation()
                        nonlin.add_input()
                    trans.setLabel(it.current_format('NL {autoindex}'))

                    self.set_input('lsnl', it, trans.Ntrue, argument_number=0)
                    self.set_output('lsnl', it, trans.Nrec)

    def get_buffers_auto(self, kobj):
        k, obj = kobj
        return k, get_buffers_graph_or_hist1(obj)

    def load_data(self):
        filename=self.vcfg['filename']
        tfile = R.TFile(filename, 'READ')
        if tfile.IsZombie():
            raise IOError('Can not read ROOT file: '+filename)

        graphs = {k: tfile.Get(v) for k, v in self.vcfg['names'].items()}

        if not all(graphs.values()):
            raise IOError('Some objects were not read from file: '+filename)

        def mult(xy):
            x, y = xy
            y*=x

        def grad(kxy):
            k, (x, y) = kxy
            return k, np.gradient(y, x[1]-x[0])

        graphs = dict(map(self.get_buffers_auto, graphs.items()))
        self.check_same_x(graphs)
        self.make_diff(graphs)
        graphs = self.supersample(graphs)
        list(map(mult, graphs.values()))
        gradients = dict(map(grad, graphs.items()))

        tfile.Close()
        return graphs, gradients

    def make_diff(self, graphs):
        names = list(self.vcfg['names'].keys())
        nom, others = names[0], names[1:]

        nominal = graphs[nom][1]

        for name in others:
            y = graphs[name][1]
            y-=nominal

    def supersample(self, graphs):
        times = self.vcfg['supersample']
        if not times or times==1:
            return graphs

        newgraphs = {}

        x = list(graphs.values())[0][0]
        nbins = len(x)-1
        newnbins = nbins*times
        newx = np.linspace(x[0], x[-1], newnbins+1)

        for name, (x,y) in graphs.items():
            fcn = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            newy = fcn(newx)
            newgraphs[name] = (newx, newy)

        return newgraphs

    def check_same_x(self, graphs):
        xcommon = None
        for name, (x, y) in graphs.items():
            if xcommon is None:
                xcommon = x
                continue

            if not np.allclose(xcommon, x, rtol=0, atol=1.e-16):
                raise self.exception('Nonlinearity curves X should be the same')

        widths = xcommon[1:]-xcommon[:-1]
        assert np.allclose(widths, widths[0], rtol=0, atol=1.e-14), "LSNL steps should be similar"

    def define_variables(self):
        par=None
        for itl in self.component_idx.iterate():
            if par is None:
                par = self.reqparameter('lsnl_weight', itl, central=1.0, fixed=True, label='Nominal nonlinearity curve weight ({autoindex})')
            else:
                par = self.reqparameter('lsnl_weight', itl, central=0.0, sigma=1.0, label='Correction nonlinearity weight for {autoindex}')

        # if 'par' in self.cfg:
            # if self.cfg.par.central!=1:
                # raise self._exception('Relative energy scale parameter should have central value of 1 by definition')

            # for it in self.detector_idx.iterate():
                # self.reqparameter('escale', it, cfg=self.cfg.par, label='Uncorrelated energy scale for {autoindex}' )

    # def interpolate(self, xy, edges):
        # x, y = xy
        # fill_ = self.cfg.get('extrapolation_strategy', 'extrapolate')
        # fcn = interp1d( x, y, kind='linear', bounds_error=False, fill_value=fill_ )
        # res = fcn( edges )
        # return res
