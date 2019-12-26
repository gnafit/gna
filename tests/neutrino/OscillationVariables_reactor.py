#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env, ExpressionsEntry
import numpy as N

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-c', '--pmnsc', action='store_true' )
opts = parser.parse_args()

ns = env.globalns
ns.defparameter('L', central=1, sigma=0.1)

from gna.parameters.oscillation import reqparameters_reactor
print('Initialize variables')
reqparameters_reactor(ns)
ns['Delta'].set(N.pi*1.5)
print('Materialize expressions')
# ns.materializeexpressions()

print('Status:')
env.globalns.printparameters(labels=True)
print()

if opts.pmnsc:
    print('Initialize variables (C)')

    checkpmns = ns('checkpmns')
    exprs = R.PMNSExpressionsC(ns=checkpmns)
    print('Materialize expressions')
    checkpmns.materializeexpressions()

ns1 = ns('weights_ae_ae')
print('Init OP ae-ae')
with ns1:
    re1=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), ns=ns1)
    print('Materialize expresions')
    ns1.materializeexpressions()
# w12 = 2 sqr c12*s12*c13*c13
# w13 = 2 sqr c12*c13*s13
# w23 = 2 sqr s12*c13*s13

ns2 = ns('weights_mu_e')
print('Init OP mu-e')
with ns2:
    re2=R.OscProbPMNSExpressions(R.Neutrino.mu(), R.Neutrino.e(), ns=ns2)
    print('Materialize expresions')
    ns2.materializeexpressions()

ns3 = ns('weights_amu_ae')
print('Init OP amu-ae')
with ns3:
    re3=R.OscProbPMNSExpressions(R.Neutrino.amu(), R.Neutrino.ae(), ns=ns3)
    print('Materialize expresions')
    ns3.materializeexpressions()

print()
print('Print final')
ns.printparameters(labels=True)

