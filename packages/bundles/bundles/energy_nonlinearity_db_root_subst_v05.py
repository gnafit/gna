from load import ROOT as R
from scipy.interpolate import interp1d
import numpy as np
import gna.constructors as C
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from gna.bundle import TransformationBundle
from tools.dictwrapper import DictWrapper

from gna.configurator import StripNestedDict
from tools.schema import Schema, And, Or, Optional, Use, haslength, isrootfile, isreadable
from typing import List, Dict, Optional as TypingOptional

class energy_nonlinearity_db_root_subst_v05(TransformationBundle):
    """Detector energy nonlinearity parametrized via few curves (Daya Bay approach)

    The bundle provides the inverse energy scale converion Evisible(Eqed)
    defined by inverting and interpolating the Eqing(Evisible) curve.
    Evisible is the total positron energy + electron mass: Evis=Ee+me.

    This plugin also keeps the HistNonlinearityB/C instance as well.

    Changes since energy_nonlinearity_db_root_subst_v04:
        - Provide relative mode, which extrapolates relative input curves and makes the gradient extrapolation linear (not constant)
        - Add an option to rescale the nominal (and pull) curves
        - Add an option to choose the order of the oprations: scale, interpolate, extrapolate, diff
    """
    _rescale_key: object = object()
    _graph_nominal: TypingOptional[List[np.ndarray]] = None
    _graph_scale:   TypingOptional[List[np.ndarray]] = None
    _graphs_all:    Dict[str, List[np.ndarray]]
    _graphs_lsnl:   Dict[str, List[np.ndarray]]
    _graphs_pull:   Dict[str, List[np.ndarray]]
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self._graphs_all={}
        self._graphs_lsnl={}
        self._graphs_pull={}

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

        order = self.vcfg['methods_order']
        if len(order)!=len(set(order)):
            raise self.exception(f'The order should not contain duplicate items: {order}')

    # Methods, applied to the LSNL curves, order sensitive
    _methodnames = [ 'scale', 'reltoabs', 'interpolate', 'extrapolate', 'diff' ]
    _validator = Schema({
            'bundle': object,
            'filename': And(isrootfile, isreadable),
            'names': Or({str: str}, And(Or([str], (str,)), Use(lambda l: {s:s for s in l}))),
            Optional('rescale_to', default=None): str,
            Optional('expose_matrix', default=False): bool,
            Optional('matrix_method', default='c'): Or('b', 'c', 'd'),
            Optional('nonlin_range', default=None): And((float,), haslength(exactly=2)),
            Optional('supersample', default=None): int,
            Optional('minor_extra', default=()): Or([str], (str,), And(str, Use(lambda s: [s]))),
            Optional('bypass_minor', default=()): Or([str], (str,), And(str, Use(lambda s: [s]))),
            Optional('verbose', default=0): Or(int, And(bool, Use(int))),
            Optional('create_weights', default=True): bool,
            Optional('relative', default=False): bool,
            Optional('extrapolation_range', default=None): And((Or(float, None),), haslength(exactly=2)),
            Optional('methods_order', default=list(_methodnames)): list(_methodnames),
            Optional('plot', default=False): bool
        })

    @staticmethod
    def _provides(cfg):
        transs = ['lsnl',
                  'lsnl_x', 'lsnl_component_y',
                  'lsnl_interpolator',
                  'lsnl_interpolator_grad',
                  'lsnl_edges',
                  'lsnl_evis',
                  'lsnl_direct', 'lsnl_direct_rel',
                  'lsnl_coarse_rel'
                  ]
        if cfg.get('matrix_method', 'c')=='c':
            transs.append('lsnl_outedges')
        pars   = ('lsnl_weight',)
        # if 'par' in cfg:
            # return ('escale',)+pars, transs

        return pars, transs

    def build(self):
        self.objects={}
        objects = DictWrapper(self.objects)
        self.load_data()

        # Correlated part of the energy nonlinearity factor
        # a weighted sum of input curves
        for i, itl in enumerate(self.component_idx.iterate()):
            name, = itl.current_values()
            if not name in self._graphs_lsnl:
                raise self._exception('The nonlinearity curve {} is not provided'.format(name))

            try:
                x, y = self._graphs_lsnl[name]
            except:
                raise Exception('Unable to get x,y for nonlinearity {}'.format(name))

            Y = C.Points(y)

            if i:
                label=itl.current_format('NL correction {autoindex}')
            else:
                label=itl.current_format('NL nominal ({autoindex})')

                X = C.Points(x)
                X.points.setLabel(label+' X')
                self.set_output('lsnl_x', None, X.single())
                objects[('curves', name, 'X')] = X

            Y.points.setLabel(label+' Y')
            self.set_output('lsnl_component_y', itl, Y.single())
            objects[('curves', name, 'Y')] = Y

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
        if not self.vcfg['relative']:
            # Use constant gradient extrapolation for linear absolute curve
            interp_deq_devis.set_underflow_strategy(R.GNA.Interpolation.Strategy.NearestEdge)
            interp_deq_devis.set_overflow_strategy(R.GNA.Interpolation.Strategy.NearestEdge)

            # otherwise
            # use linear gradient extrapolation for quadratic absolute curve (for linear relative curve)

        matrix_method = self.vcfg['matrix_method']
        # Interp_interpolators
        # interp_evis acts similarly to interp_invers, but expects different interpolation points
        #
        # x, y     -> interp_direct  : interp_direct(bin edges)
        # y, x     -> interp_inverse : interp_direct(bin edges)
        # y, x     -> interp_evis    : interp_evis(integration points of Eq)
        # x, dy/dx -> interp_grad    : interp_grad(Evis(integration points of Eq))
        #
        #            0     1   2           3                      4
        # Arguments: Evis, Eq, E edges in, E integration points[, E edges out]
        # Distribution:
        #   Interpolator     Xorig   Yorig      Xnew
        #   interp_direct    Evis    Eq         E edges in
        #   interp_inverse   Eq      Evis       E edges in (b, c) or out (d)
        #   interp_evis      Eq      Evis       E integration points
        #   interp_grad      Evis    dEq/dEvis  interp_evis: Evis(Eq=E integration points)
        #
        evis_inputs_coarse =                    (interp_direct.insegment.edges, interp_direct.interp.x)
        evis_inputs_coarse = evis_inputs_coarse+(interp_inverse.interp.y,)
        evis_inputs_coarse = evis_inputs_coarse+(interp_evis.interp.y,)
        eq_input_coarse    =                 (interp_direct.interp.y,)
        eq_input_coarse    = eq_input_coarse+(interp_inverse.insegment.edges, interp_inverse.interp.x)
        eq_input_coarse    = eq_input_coarse+(interp_evis.insegment.edges, interp_evis.interp.x)
        e_edges_input      = (interp_direct.insegment.points, interp_direct.interp.newx)
        e_intpoints_input  = (interp_evis.insegment.points, interp_evis.interp.newx)
        e_edges_out        = e_edges_input+(interp_inverse.insegment.points, interp_inverse.interp.newx)
        if matrix_method!='d':
            e_edges_input=e_edges_out

        #
        # Convert the output of direct interpolation to rel/abs
        #
        lsnl_direct_abs = interp_direct.interp.interp
        lsnl_direct_rel_t = C.Ratio(labels='LSNL direct rel')
        objects['ratio_direct']=lsnl_direct_rel_t
        lsnl_direct_rel = lsnl_direct_rel_t.ratio.ratio
        lsnl_direct_abs >> lsnl_direct_rel_t.ratio.top
        e_edges_input += lsnl_direct_rel_t.ratio.bottom,

        #
        # Convert
        #
        eq_abs_input_t = C.Ratio(labels='LSNL abs to rel')
        objects['ratio_coarse'] = eq_abs_input_t
        self.set_output('lsnl_coarse_rel', None, eq_abs_input_t.ratio.ratio)
        evis_inputs_coarse += eq_abs_input_t.ratio.bottom,
        eq_input_coarse += eq_abs_input_t.ratio.top,

        #
        # Set inputs for regular interpolators
        #
        self.set_input('lsnl_interpolator', None, evis_inputs_coarse, argument_number=0)
        self.set_input('lsnl_interpolator', None, eq_input_coarse, argument_number=1)
        self.set_input('lsnl_interpolator', None, e_edges_input, argument_number=2)
        self.set_input('lsnl_interpolator', None, e_intpoints_input, argument_number=3)
        if matrix_method=='d':
            self.set_input('lsnl_interpolator', None, e_edges_out, argument_number=4)

        #
        # Set inputs for gradient interpolator
        #
        arg0=interp_deq_devis.interp.y
        self.set_input('lsnl_interpolator_grad', None, arg0, argument_number=0)
        arg0=(interp_deq_devis.insegment.edges, interp_deq_devis.interp.x)
        self.set_input('lsnl_interpolator_grad', None, arg0, argument_number=1)

        #
        # Set regular outputs
        #
        self.set_output('lsnl_direct',  None, lsnl_direct_abs)
        self.set_output('lsnl_direct_rel',  None, lsnl_direct_rel)
        self.set_output('lsnl_inverse', None, interp_inverse.interp.interp)
        self.set_output('lsnl_evis',    None, interp_evis.interp.interp)
        self.set_output('lsnl_interpolator_grad', None, interp_deq_devis.interp.interp)

        #
        # Bind gradient interpolator
        #
        interp_evis.interp.interp >> (interp_deq_devis.insegment.points, interp_deq_devis.interp.newx)

        #
        # Build matrices
        #
        expose_matrix = R.GNA.DataPropagation.Propagate if self.vcfg['expose_matrix'] else R.GNA.DataPropagation.Ignore
        nonlin_range = self.vcfg['nonlin_range']
        bypass_minor = self.vcfg['bypass_minor']
        verbosity = self.vcfg['verbose']

        provide_outedges = False
        if matrix_method=='d':
            def make_matrix_lsnl(**kwargs):
                return R.HistNonlinearityD(expose_matrix, **kwargs)
        elif matrix_method=='c':
            provide_outedges = True
            def make_matrix_lsnl(**kwargs):
                return R.HistNonlinearityC(expose_matrix, **kwargs)
        elif matrix_method=='b':
            def make_matrix_lsnl(**kwargs):
                return R.HistNonlinearityB(expose_matrix, **kwargs)
        else:
            assert False

        for i, itd in enumerate(self.detector_idx.iterate()):
            """Finally, original bin edges multiplied by the correction factor"""
            """Construct the nonlinearity calss"""
            with self.namespace:
                nonlin = make_matrix_lsnl(labels=itd.current_format('NL matrix {autoindex}'))
            if nonlin_range:
                nonlin.set_range(*nonlin_range)

            objects[('nonlinearity',)+itd.current_values()] = nonlin

            if matrix_method=='d':
                lsnl_direct_abs  >> nonlin.matrix.EdgesInModified
                interp_inverse.interp.interp >> nonlin.matrix.EdgesOutBackwardProjection
                self.set_input('lsnl_edges', itd, nonlin.matrix.EdgesIn, argument_number=0)
                self.set_input('lsnl_edges', itd, nonlin.matrix.EdgesOut, argument_number=1)
            else:
                lsnl_direct_abs  >> nonlin.matrix.EdgesModified
                interp_inverse.interp.interp >> nonlin.matrix.BackwardProjection
                self.set_input('lsnl_edges', itd, nonlin.matrix.Edges, argument_number=0)

            self.set_output('lsnl_matrix', itd, nonlin.matrix.FakeMatrix)
            if provide_outedges:
                self.set_output('lsnl_outedges', itd, nonlin.matrix.OutEdges)

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

    def get_buffers_auto(self, obj) -> List[np.ndarray]:
        return list(get_buffers_graph_or_hist1(obj))

    def load_data(self) -> None:
        self.read_graphs()
        self.check_same_x()

        if self.vcfg['plot']:
            plot = self._plot
            plot('Initial')
        else:
            plot = lambda title: None

        methods = {
            'scale': self._method_scale,
            'reltoabs': self._method_reltoabs,
            'interpolate': self._method_interpolate,
            'extrapolate': self._method_extrapolate,
            'diff': self._method_diff,
        }

        for methodname in self.vcfg['methods_order']:
            method = methods[methodname]
            method()
            plot(methodname)

        if self.vcfg['plot']:
            from matplotlib import pyplot as plt
            plt.show()

    def _method_scale(self):
        if self._graph_scale is None:
            return

        _, rescaleto_y = self._graph_scale
        _, nominal_y = self._graph_nominal

        correction = rescaleto_y/nominal_y
        nominal_y[:] = rescaleto_y

        for (_, pull_y) in self._graphs_pull.values():
            pull_y[:] *= correction

    def _method_diff(self) -> None:
        _, nominal_y = self._graph_nominal

        for (_, pull_y) in self._graphs_pull.values():
            pull_y-=nominal_y

    def _method_reltoabs(self) -> None:
        for (graphx, graphy) in self._graphs_all.values():
            graphy[:]*=graphx

    def _method_interpolate(self) -> None:
        times = self.vcfg['supersample']
        if not times or times==1:
            return

        newgraphs = {}

        xcoarse, _ = self._graph_nominal

        newshape = (len(xcoarse)-1)*times + 1
        xfine = np.linspace(xcoarse[0], xcoarse[-1], newshape)

        for xypair in self._graphs_all.values():
            xcoarse, ycoarse = xypair
            fcn = interp1d(xcoarse, ycoarse, kind='cubic', bounds_error=False, fill_value='extrapolate')
            yfine = fcn(xfine)

            xypair[0] = xfine.copy()
            xypair[1] = yfine

    def _method_extrapolate(self) -> None:
        extrap = self.vcfg['extrapolation_range']
        if not extrap:
            return

        extrap_left, extrap_right = extrap
        if extrap_left is None and extrap_right is None:
            return

        xbounded, _ = self._graph_nominal
        if extrap_left is not None:
            xleft = np.arange(extrap_left, xbounded[0], xbounded[1]-xbounded[0])
        else:
            xleft = None
        if extrap_right is not None:
            stepright = xbounded[-1] - xbounded[-2]
            xright = np.arange(xbounded[-1], extrap_right+stepright*1.e-6, stepright)
        else:
            xright = None

        for xypair in self._graphs_all.values():
            xbounded, ybounded = xypair
            fcn = interp1d(xbounded, ybounded, kind='linear', bounds_error=False, fill_value='extrapolate')

            xstack, ystack = [], []
            if xleft is not None:
                yleft = fcn(xleft)
                xstack.append(xleft)
                ystack.append(yleft)
            xstack.append(xbounded)
            ystack.append(ybounded)
            if xright is not None:
                yright = fcn(xright)
                xstack.append(xright)
                ystack.append(yright)

            xnew = np.concatenate(xstack)
            ynew = np.concatenate(ystack)

            xypair[0] = xnew
            xypair[1] = ynew

    def _plot(self, title: str) -> None:
        from matplotlib import pyplot as plt

        for name, (x, y) in self._graphs_all.items():
            if not isinstance(name, str):
                name = 'scale'

            if plt.fignum_exists(name):
                fig = plt.figure(name)
                ax = fig.axes[0]
            else:
                plt.figure(name)
                ax = plt.subplot(111, xlabel='E, MeV', ylabel='', title=f'LSNL part: {name}')

            ax.plot(x, y, 'o', markerfacecolor='none', markersize=0.3, alpha=0.5, label=title)
            ax.legend()

    def read_graphs(self) -> None:
        filename=self.vcfg['filename']
        tfile = R.TFile(filename, 'READ')
        if tfile.IsZombie():
            raise IOError('Can not read ROOT file: '+filename)

        keynamepairs = list(self.vcfg['names'].items())
        if not keynamepairs:
            raise self.exception('No configuration for LSNL curves provided')

        rescale_to_name = self.vcfg['rescale_to']
        if rescale_to_name:
            keynamepairs.append((self._rescale_key, rescale_to_name))

        if self._graphs_all:
            raise self.exception('Graphs already loaded')

        for key, name in keynamepairs:
            obj = tfile.Get(name)
            if not obj:
                raise self.exception(f'Unable to read "{key}" from "{filename}"')

            xy = self.get_buffers_auto(obj)

            targets = (self._graphs_all,)
            if isinstance(key, str):
                targets += (self._graphs_lsnl,)

                if self._graph_nominal is None:
                    self._graph_nominal = xy
                else:
                    targets += (self._graphs_pull,)
            else:
                assert self._graph_scale is None
                self._graph_scale = xy

            for target in targets:
                target[key] = xy

        tfile.Close()

    def check_same_x(self) -> None:
        xcommon = None
        for name, (x, _) in self._graphs_all.items():
            if xcommon is None:
                xcommon = x
                continue

            if not np.allclose(xcommon, x, rtol=0, atol=1.e-16):
                raise self.exception('Nonlinearity curves X should be the same')

        widths = xcommon[1:]-xcommon[:-1]
        assert np.allclose(widths, widths[0], rtol=0, atol=1.e-14), "LSNL steps should be similar"

    def define_variables(self):
        if not self.vcfg['create_weights']:
            return

        par=None
        for itl in self.component_idx.iterate():
            if par is None:
                par = self.reqparameter('lsnl_weight', itl, central=1.0, fixed=True, label='Nominal nonlinearity curve weight ({autoindex})')
            else:
                par = self.reqparameter('lsnl_weight', itl, central=0.0, sigma=1.0, label='Correction nonlinearity weight for {autoindex}')
