from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import NestedDict
from collections.abc import Mapping

class integral_2d1d_v06(TransformationBundle):
    """2d integral based on Integrator21GL v06

    Changes since integral_2d1d_v05:
        - Add an option to override bundle index

    Configuration options:
        - instances   - dictionary of (name, label) pairs for instances
                        { 'name1': 'label1', ... }
                        alternativetly, a a dictionary with extra options may be passed instead of label:
                        { 'name1': {'label': 'label1', 'index': ('i1', 'i2')}, ... }
                        Extra options:
                            * 'index' - use subset of indices
        - xedgescfg - a list of 3-tuples:
                      [
                          (left1, step1, order1),
                          (left2, step2, order2),
                          ...
                          (last, None, None),
                      ]
                      an interval [left1, left2) will be filled with points with step1 (similar to arange),
                      each interval will have order=order1

        - yorder    - number of points in bin on axis Y, integer.
        - variables - variable names (array of 2 strings).

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
        self.init()

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

        # try:
            # self.edges = np.ascontiguousarray(self.cfg.edges, dtype='d')
        # except:
            # raise self.exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        # try:
            # self.xorders = np.ascontiguousarray(self.cfg.xorders, dtype='P')
        # except:
            # raise self.exception('Invalid xorders definition: {!r}'.format(self.cfg.xorders))

        if len(self.cfg.variables)!=2:
            raise self.exception('Two vairables should be provided')

    def init(self):
        edges_list = []
        orders_list = []

        edgescfg = self.cfg.xedgescfg
        for (start, step, orders), (stop, _, _) in zip(edgescfg[:-1], edgescfg[1:]):
            cedges = np.arange(start, stop, step, dtype='d')
            edges_list.append(cedges)
            orders_list.append(np.ones_like(cedges, dtype='i')*orders)

        edges_list.append([stop])

        self.xedges=np.concatenate(edges_list)
        self.xorders=np.concatenate(orders_list)

    def build(self):
        if self.xorders.size>1:
            if self.xorders.size+1 != self.xedges.size:
                raise self.exception('Incompartible edges and xorders definition:\n    {!r}\n    {!r}'.format(self.xedges, self.xorders))
            self.integrator = R.Integrator21GL(self.xedges.size-1, self.xorders, self.xedges, self.cfg.yorder, -1.0, 1.0)
        else:
            self.integrator = R.Integrator21GL(self.xedges.size-1, int(self.xorders[0]), self.xedges, self.cfg.yorder, -1.0, 1.0)
        self.integrator.points.setLabel('GL sampler (2d)')

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
            subindex = None
            if isinstance(label, (Mapping, NestedDict)):
                subindex = label.get('index')
                label = label.get('label')

            if label is None:
                label = '{name} {autoindex} (GL)'

            nidx = self.nidx
            if subindex is not None:
                nidx = self.nidx.get_subset(subindex)
            for it in nidx:
                if needadd:
                    hist = self.integrator.add_transformation()
                needadd=True

                hist.setLabel(it.current_format(label, name='Integral'))

                self.set_input(name, it, self.integrator.add_input(), argument_number=0)
                self.set_output(name, it, hist.outputs.back())

    def define_variables(self):
        pass
