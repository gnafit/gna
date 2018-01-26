# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from collections import OrderedDict
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from constructors import Histogram
from gna.configurator import NestedDict, uncertain
from converters import convert

from gna.bundle import *

class bkg_weighted_hist_v01(TransformationBundle):
    name = 'bkg_weighted_hist'

    formats = dict(
            num  = '{}_num',
            )
    def __init__(self, **kwargs):
        super(bkg_weighted_hist_v01, self).__init__( **kwargs )

        self.spectra = execute_bundle( cfg=self.cfg.spectra, storage=self.storage )
        self.namespaces = self.spectra.namespaces

        for site, detectors in self.cfg.parent().parent()['groups'].items():
            sitens = self.common_namespace(site)
            sitens['detectors'] = {}
            for det in detectors:
                detns = self.common_namespace(det)
                detns['site'] = self.common_namespace(site)
                detns['exp'] = self.common_namespace
                detns.reqparameter('livetime', central=10, sigma=0.1, fixed=True)
                detns['detectors'] = { det: detns }

                sitens['detectors'][det]=detns

        self.cfg.setdefault( 'name', self.cfg.parent_key() )
        print( 'Executing:\n', str(self.cfg), sep='' )

    def build(self):
        for ns in self.iterate_namespaces():
            # ws = R.WeightedSum()
            print(ns.name)

    def define_variables(self):
        pitems = None
        if len(self.cfg.formula)>1:
            pitems = convert( self.cfg.formula, 'stdvector' )

        self.products=[]
        numname = self.formats['num'].format( self.cfg.name )
        for ns in self.namespaces:
            bindings = {}
            for fullitem in self.cfg.formula:
                if '.' in fullitem:
                    item = fullitem.split('.')[-1]
                else:
                    if not fullitem in self.cfg:
                        continue
                    item = fullitem
                num = self.cfg[item]

                if isinstance( num, uncertain ):
                    cnum = num
                else:
                    cnum = num[ns.name]
                bindings[fullitem] = ns.reqparameter( item.format( self.cfg.name ), cnum )

            for detns in ns['detectors'].values():
                if pitems:
                    with detns:
                        vp = R.VarProduct( pitems, numname, ns=detns, bindings=bindings )
                        detns[numname].get()
                        self.products.append( vp )
                # else:
                    # ns[numname] = R.Variable('double')( numname, ns[formula[0]].getVariable() )
                    # # ns.defparameter( numname, target=formula[0] )

