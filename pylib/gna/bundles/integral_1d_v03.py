from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict
from collections.abc import Mapping

class integral_1d_v03(TransformationBundle):
    """1d integral bundle v03

    Integral_1d bundle. Creates necessary transformations to integrate 1d function into 1d histogram
    with predefined bins using Guss-Legendre quadrature.

    Changes since v02:
      - Switch to IntegratorGL (deprecate GaussLegendre)
      - Support instances
      - Provide all the variables for x: centers, edges, histogram

    Configuration:
        kinint = NestedDict(
            bundle   = 'integral_1d_v01',
            variable = 'evis',                                # - the bin edges
            edges    = N.linspace(0.0, 12.0, 241, dtype='d'), # - the integration order for each bin (or fo all of the bins) (Gauss-Legendre)
            orders   = 3,                                     # - this line says that the bundle will create 'evis' output in addition to 'kinint'
            ),
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__( self, *args, **kwargs )
        self.check_cfg()

    @staticmethod
    def _provides(cfg):
        var0 = cfg.variable
        variables = (var0, var0+'_centers', var0+'_edges', var0+'_hist')
        names = tuple(cfg['instances'].keys())
        return (), names+variables

    def check_cfg(self):
        """Checks the configuration consistency"""

        try:
            self.edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        except:
            raise self.exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        # Read the integration orders
        try:
            self.orders = N.ascontiguousarray(self.cfg.orders, dtype='P')
        except:
            raise self.exception('Invalid orders definition: {!r}'.format(self.cfg.orders))

    def build(self):
        integrator = self.cfg.get('integrator', 'gl')
        integrators = { 'gl': R.IntegratorGL, 'rect': R.IntegratorRect }
        try:
            Integrator = integrators[integrator]
        except KeyError:
            raise self.exception('Invalid integrator {}'.format(integrator))

        if self.orders.size>1:
            if self.orders.size+1 != self.edges.size:
                raise self.exception('Incompartible edges and orders definition:\n    {!r}\n    {!r}'.format(self.edges, self.orders))
            # In case orders is an array, pass it as an array
            self.integrator = Integrator(self.edges.size-1, self.orders, self.edges)
        else:
            # Or pass it as an integer
            self.integrator = Integrator(self.edges.size-1, int(self.orders[0]), self.edges)

        self.integrator.points.x.setLabel(self.cfg.variable)
        self.integrator.points.xedges.setLabel('%s edges'%self.cfg.variable)
        self.integrator.points.xcenters.setLabel('{} bin centers'.format(self.cfg.variable))

        # Register the outputs:
        #   - the points to compute function on
        #   - the bin edges
        self.set_output(self.cfg.variable, None, self.integrator.points.x)
        self.set_output('{}_edges'.format(self.cfg.variable), None, self.integrator.points.xedges)
        self.set_output('{}_centers'.format(self.cfg.variable), None, self.integrator.points.xcenters)
        self.set_output('{}_hist'.format(self.cfg.variable), None, self.integrator.points.xhist)

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
