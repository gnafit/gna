# Reimplementation of integral_2d1d_v01 for up to date bundles
# Reimplementation of integral_2d1d_v02 with Integrator21GL instead of legacy GaussLegendre
# Reimplementation of integral_2d1d_v03 with option for Integrator2Rect
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict

class integral_2d1d_v05(TransformationBundle):
    """2d integral based on Integrator21GL v04 and Integrator2Rect

    Changes since integal_2d1d_v04:
        -Added 'mode' option for rectangular integrator

    Changes since integral_2d1d_v03:
        - Added 'instances' option

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
                        alternativetly, a a dictionary with extra options may be passed instead of label:
                        { 'name1': {'label': 'label1', 'noindex': True}, ... }
                        Extra options:
                            * 'noindex' - disable indices
        - edges     - array with output histogram edges
        - xorders   - number of points in each bin on axis X, integer or array or integers.
        - yorder    - number of points in bin on axis Y, integer.
        - variables - variable names (array of 2 strings).
        - mode      - integrator mode (GL, rect, rect_left, rect_right)

    Predefined names:
        - variables[0] - array with points to integrate over variable[0]
        - variables[1] - array with points to integrate over variable[1]
        - variables[0]+'_edges' - array with bin edges for variable[0]
        - variables[0]+'_centers' - array with bin centers for variable[0]
        - variables[0]+'_hist' - histogram with bin edges for variable[0]
        - variables[i]+'_mesh' - 2d mesh for variable[i] (as in numpy.meshgrid)

        (may be configured via 'names' option of a bundle)
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__( self, *args, **kwargs )
        self.check_cfg()

    @staticmethod
    def _provides(cfg):
        var0, var1 = cfg.variables
        variables = (var0, var1, var0+'_centers', var0+'_edges', var0+'_hist', var0+'_mesh', var1+'_mesh')
        names = tuple(cfg['instances'].keys())
        return (), names+variables

    def check_cfg(self):
        if not 'name' in self.cfg:
            pkey = self.cfg.parent_key()
            if not pkey:
                raise self.exception('"name" option is not provided')
            self.cfg.name = pkey

        try:
            self.edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        except:
            raise self.exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        try:
            self.xorders = N.ascontiguousarray(self.cfg.xorders, dtype='P')
        except:
            raise self.exception('Invalid xorders definition: {!r}'.format(self.cfg.xorders))

        if len(self.cfg.variables)!=2:
            raise self.exception('Two vairables should be provided')

        if not self.cfg.mode:
            self.mode = 'GL'
        elif self.cfg.mode not in ['GL', 'rect', 'rect_left', 'rect_right']:
            raise self.exception('Invalid integrator mode')
        else:
            self.mode = self.cfg.mode

        if self.cfg.mode != 'GL':
            self.yedges = N.ascontiguousarray([-1.0, 1.0], dtype='d')
            

        

    def build(self):
        if self.xorders.size>1:
            if self.xorders.size+1 != self.edges.size:
                raise self.exception('Incompartible edges and xorders definition:\n    {!r}\n    {!r}'.format(self.edges, self.xorders))
            if self.mode == 'GL':
                self.integrator = R.Integrator21GL(self.edges.size-1, self.xorders, self.edges, self.cfg.yorder, -1.0, 1.0)
            else:
                self.integrator = R.Integrator2Rect(self.edges.size-1, self.xorders, self.edges, self.yedges.size-1, self.cfg.yorder, self.yedges)
        else:
            if self.mode == 'GL':
                self.integrator = R.Integrator21GL(self.edges.size-1, int(self.xorders[0]), self.edges, self.cfg.yorder, -1.0, 1.0)
                self.integrator.points.setLabel('GL sampler (2d)')
            else:
                self.integrator = R.Integrator2Rect(self.edges.size-1, int(self.xorders[0]), self.edges, self.yedges.size-1, self.cfg.yorder, self.yedges, self.mode)
                self.integrator.points.setLabel('Rectangular sampler (2d)')

        self.integrator.points.x.setLabel(self.cfg.variables[0])
        self.integrator.points.xedges.setLabel('%s edges'%self.cfg.variables[0])
        self.integrator.points.xcenters.setLabel('{} bin centers'.format(self.cfg.variables[0]))
        self.integrator.points.y.setLabel(self.cfg.variables[1])

        self.set_output(self.cfg.variables[0],            None, self.integrator.points.x)
        self.set_output('{}_edges'.format(self.cfg.variables[0]), None, self.integrator.points.xedges)
        self.set_output('{}_centers'.format(self.cfg.variables[0]),  None, self.integrator.points.xcenters)
        self.set_output('{}_hist'.format(self.cfg.variables[0]),  None, self.integrator.points.xhist)
        self.set_output('{}_mesh'.format(self.cfg.variables[0]),  None, self.integrator.points.xmesh)
        self.set_output('{}_mesh'.format(self.cfg.variables[1]),  None, self.integrator.points.ymesh)
        self.set_output(self.cfg.variables[1],            None, self.integrator.points.y)

        hist = self.integrator.hist

        instances = self.cfg['instances']
        needadd = False
        for name, label in instances.items():
            noindex = False
            if isinstance(label, (dict, NestedDict)):
                noindex = label.get('noindex')
                label = label.get('label')

            if label is None:
                label = '{name} {autoindex} (GL)'

            for it in self.nidx:
                if needadd:
                    hist = self.integrator.add_transformation()
                needadd=True

                if noindex:
                    hist.setLabel(label)
                    self.set_input(name, None, self.integrator.add_input(), argument_number=0)
                    self.set_output(name, None, hist.outputs.back())
                    break

                hist.setLabel(it.current_format(label, name='Integral'))

                self.set_input(name, it, self.integrator.add_input(), argument_number=0)
                self.set_output(name, it, hist.outputs.back())

    def define_variables(self):
        pass
