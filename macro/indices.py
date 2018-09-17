#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gna.expression import *

# speci = IWeighted( 'Isotoper', [['ffrac', 'i', 'r'], ['power', 'r']], ('Isotope', 'i') )
# offeq = IWeighted( 'Offeqr',   [['ffrac', 'i', 'r'], ['offeq_norm', 'i', 'r'], ['power', 'r']], ('Offeq', 'i') )
# snf   = IWeighted( 'Snf',      [['snf_norm', 'r'], ['power', 'r']], ('Snf', 'i') )
# comp0 = IWeighted( 'Comp0',    [['oscprobw0',]], ('OscProbItem0',) )
# compi = IWeighted( 'CompI',    [['oscprobw', 'c']], ('OscProbItem', 'c') )
# integr = Indexed( 'Integrate' )
# xsec   = Indexed( 'Xsec' )
# jac    = Indexed( 'Jac' )
# dnorm  = Indexed( 'detnorm', 'd' )
# baseline  = Indexed( 'baseline', 'd', 'r' )
# rnorm  = Indexed( 'reacnorm', 'r' )
# print( speci )
# print( offeq )
# print( snf )
# print( comp0 )
# print( compi )
# print()

# spec=Sum('spec', speci, offeq, snf)
# comp=Sum('comp', comp0, compi)
# ps = Product( 'prod', dnorm, rnorm, baseline, integr, xsec, jac, comp, spec )

# print( 'Raw' )
# print( ps )

# print( 'Extract' )
# ps.extract()
# print(ps)

# print( 'Arrange' )
# op=ps.open()
# print(op)
