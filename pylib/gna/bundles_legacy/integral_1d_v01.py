# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *

class integral_1d_v01(TransformationBundleLegacy):
    # @brief 1d integral bundle
    #
    # Integral_1d bundle. Creates necessary transformations to integrate 1d function into 1d histogram
    # with predefined bins using Guss-Legendre quadrature.
    #
    # Configuration:
    #     @code{.py}
    #     kinint = NestedDict(
    #         # bundle name
    #         bundle   = 'integral_1d_v01',
    #         # The following lines are the bundle options:
    #         # - the integration variable name
    #         variable = 'evis',
    #         # - the bin edges
    #         edges    = N.linspace(0.0, 12.0, 241, dtype='d'),
    #         # - the integration order for each bin (or fo all of the bins) (Gauss-Legendre)
    #         orders   = 3,
    #         # - this line says that the bundle will create 'evis' output in addition to 'kinint'
    #         provides = [ 'evis' ]
    #         ),
    #     @endcode
    def __init__(self, *args, **kwargs):
        TransformationBundleLegacy.__init__( self, *args, **kwargs )
        self.check_cfg()

    def check_cfg(self):
        """Checks the configuration consistency"""

        # Get the parent key and use as output name in case it is not provided
        if not 'name' in self.cfg:
            pkey = self.cfg.parent_key()
            if not pkey:
                raise Exception('"name" option is not provided for integral_1d_v01')
            self.cfg.name = pkey

        # Initialize indices, needed to provide multiple outputs if needed
        self.idx = self.cfg.indices
        from gna.expression import NIndex
        if not isinstance(self.idx, NIndex):
            self.idx = NIndex(fromlist=self.cfg.indices)
        try:
            self.edges = N.ascontiguousarray(self.cfg.edges, dtype='d')
        except:
            raise Exception('Invalid binning definition: {!r}'.format(self.cfg.edges))

        # Read the integration orders
        try:
            self.orders = N.ascontiguousarray(self.cfg.orders, dtype='P')
        except:
            raise Exception('Invalid orders definition: {!r}'.format(self.cfg.orders))

    def build(self):
        """Build tha actual bundle: the computational chain"""

        # Create the integrator instance. Integrator will provide the points and weights, needed to
        # compute the integral. The argumentas are:
        # 1. Bin edges.
        # 2. Integration orders.
        # 3. Number of bins.
        if self.orders.size>1:
            if self.orders.size+1 != self.edges.size:
                raise Exception('Incompartible edges and orders definition:\n    {!r}\n    {!r}'.format(self.edges, self.orders))
            # In case orders is an array, pass it as an array
            self.integrator = R.GaussLegendre(self.edges, self.orders, self.edges.size-1)
        else:
            # Or pass it as an integer
            self.integrator = R.GaussLegendre(self.edges, int(self.orders[0]), self.edges.size-1)
        self.integrator.points.setLabel('Gauss-Legendre 1d')

        # Assign labels to the outputs of the integrator. Needed for graphviz.
        self.integrator.points.x.setLabel(self.cfg.variable)
        self.integrator.points.xedges.setLabel('%s edges'%self.cfg.variable)

        # Register the outputs:
        #   - the points to compute function on
        #   - the bin edges
        self.set_output(self.integrator.points.x,      self.cfg.variable)
        self.set_output(self.integrator.points.xedges, '%s_edges'%self.cfg.variable)

        # Provide the histograms for each permutation of the indices:
        for i, it in enumerate(self.idx.iterate()):
            # Create a histogram instance, that will convert an integrator output to the histogram
            hist = R.GaussLegendreHist(self.integrator)
            # label it (graphviz)
            hist.hist.setLabel(it.current_format(name='hist'))

            # register the input and output
            self.set_input( hist.hist.f,    self.cfg.name, it, clone=0)
            self.set_output(hist.hist.hist, self.cfg.name, it)

            # The integration procedure is a function call. Let us assume, that name is 'integral'
            # and variable is 'x'.
            # Then:
            #   integral(function(x()))
            # will copmute the integral. It works as follows:
            # 1. The array of all points 'x' needed to compute the integral is provided by this
            # bundle. It is passed as an input to some other function 'function'. The result is a
            # function computed for each value in 'x'. The output of 'function(...)' is passed as an
            # input of 'integral(...)' that converts an array to a proper histogram.

    def define_variables(self):
        """Defines the variables (no variables are needed for integral)"""
        pass

