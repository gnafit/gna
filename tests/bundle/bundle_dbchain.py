#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
R.GNAObject
from gna.bundle.detector_dbchain import detector_dbchain
from gna.env import env

parname = 'DiagScale'
par = env.defparameter( parname,  central=1.0, relsigma=0.1 )
par1 = env.defparameter( 'ad1.'+parname,  central=1.5, relsigma=0.1 )
par2 = env.defparameter( 'ad2.'+parname,  central=0.5, relsigma=0.1 )

names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3'  ]
pars = [ env.defparameter( 'weight_'+names[0], central=1.0, sigma=0.0, fixed=True ) ]
for name in names[1:]:
    par = env.defparameter( 'weight_'+name, central=0.0, sigma=1.0 )
    pars.append( par )

escale1 = env.defparameter( 'ad1.escale', central=1.0, sigma=0.02*0.01 )
escale2 = env.defparameter( 'ad2.escale', central=1.0, sigma=0.02*0.01 )

(esmear,), _ = detector_iav_from_file( 'output/detector_iavMatrix_P14A_LS.root', 'iav_matrix', ndiag=1, parname=parname )

names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3'  ]
filename = 'output/detector_nl_consModel_450itr.root'
(nonlin,), storage = detector_nl_from_file( filename, names, edges=points.points, debug=True)
