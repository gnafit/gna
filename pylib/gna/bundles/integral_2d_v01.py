# Reimplementation of integral_2d1d_v01 for up to date bundles
# Reimplementation of integral_2d1d_v02 with Integrator21GL instead of legacy GaussLegendre

from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict
from collections.abc import Mapping

class integral_2d_v01(TransformationBundle):
    """2d integral based on Integrator2GL v01

    Changes since integral_2d1d_v04:
        - Added full 2d integration

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
                        alternativetly, a a dictionary with extra options may be passed instead of label:
                        { 'name1': {'label': 'label1', 'noindex': True}, ... }
                        Extra options:
                            * 'noindex' - disable indices
        - xedges    - array with output histogram X edges
        - xorders   - number of points in each bin on axis X, integer or array or integers.
        - yedges    - array with output histogram Y edges
        - yorders   - number of points in bin on axis Y, integer.
        - variables - variable names (array of 2 strings).

    Predefined names:
        - variables[i] - array with points to integrate over variable[i]
        - variables[i]+'_edges' - array with bin edges for variable[i]
        - variables[i]+'_centers' - array with bin centers for variable[i]
        - variables[i]+'_hist' - histogram with bin edges for variable[i]
        - variables[i]+'_mesh' - 2d mesh for variable[i] (as in numpy.meshgrid)

        (may be configured via 'names' option of a bundle)
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__( self, *args, **kwargs )
        self.check_cfg()

    @staticmethod
    def _provides(cfg):
        variables = tuple(var+suffix for var in cfg.variables for suffix in ['', '_centers', '_edges', '_hist', '_mesh'])
        names = tuple(cfg['instances'].keys())
        return (), names+variables

    def check_cfg(self):
        try:
            self.xedges = N.ascontiguousarray(self.cfg.xedges, dtype='d')
        except:
            raise self.exception('Invalid binning definition: {!r}'.format(self.cfg.xedges))

        try:
            self.yedges = N.ascontiguousarray(self.cfg.yedges, dtype='d')
        except:
            raise self.exception('Invalid binning definition: {!r}'.format(self.cfg.yedges))

        try:
            self.xorders = self.cfg.xorders
            if not isinstance(self.xorders, int):
                self.xorders = N.ascontiguousarray(self.cfg.xorders, dtype='P')

                if self.xorders.size+1 != self.xedges.size:
                    raise self.exception('Incompartible xedges and xorders definition:\n'
                                         '    {!r}\n    {!r}'.format(self.xedges, self.xorders))
        except:
            raise self.exception('Invalid xorders definition: {!r}'.format(self.cfg.xorders))

        try:
            self.yorders = self.cfg.yorders
            if not isinstance(self.yorders, int):
                self.yorders = N.ascontiguousarray(self.cfg.yorders, dtype='P')

                if self.yorders.size+1 != self.yedges.size:
                    raise self.exception('Incompartible yedges and yorders definition:\n'
                                         '    {!r}\n    {!r}'.format(self.yedges, self.yorders))
        except:
            raise self.exception('Invalid yorders definition: {!r}'.format(self.cfg.yorders))

        if len(self.cfg.variables)!=2:
            raise self.exception('Two vairables should be provided')

    def build(self):
        self.integrator = R.Integrator2GL(self.xedges.size-1, self.xorders, self.xedges,
                                          self.yedges.size-1, self.yorders, self.yedges)
        self.integrator.points.setLabel('GL sampler (2d)')

        self.integrator.points.x.setLabel(self.cfg.variables[0])
        self.integrator.points.xedges.setLabel('%s edges'%self.cfg.variables[0])
        self.integrator.points.xcenters.setLabel('{} bin centers'.format(self.cfg.variables[0]))

        self.integrator.points.y.setLabel(self.cfg.variables[1])
        self.integrator.points.yedges.setLabel('%s edges'%self.cfg.variables[1])
        self.integrator.points.ycenters.setLabel('{} bin centers'.format(self.cfg.variables[1]))

        self.set_output(self.cfg.variables[0],                      None, self.integrator.points.x)
        self.set_output('{}_edges'.format(self.cfg.variables[0]),   None, self.integrator.points.xedges)
        self.set_output('{}_centers'.format(self.cfg.variables[0]), None, self.integrator.points.xcenters)
        self.set_output('{}_hist'.format(self.cfg.variables[0]),    None, self.integrator.points.xhist)
        self.set_output('{}_mesh'.format(self.cfg.variables[0]),    None, self.integrator.points.xmesh)

        self.set_output(self.cfg.variables[1],                      None, self.integrator.points.y)
        self.set_output('{}_edges'.format(self.cfg.variables[1]),   None, self.integrator.points.yedges)
        self.set_output('{}_centers'.format(self.cfg.variables[1]), None, self.integrator.points.ycenters)
        self.set_output('{}_hist'.format(self.cfg.variables[1]),    None, self.integrator.points.yhist)
        self.set_output('{}_mesh'.format(self.cfg.variables[1]),    None, self.integrator.points.ymesh)

        hist = self.integrator.hist

        instances = self.cfg['instances']
        needadd = False
        for name, label in instances.items():
            noindex = False
            if isinstance(label, (Mapping, NestedDict)):
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
