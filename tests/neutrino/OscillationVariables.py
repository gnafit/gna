#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env, ExpressionsEntry
import numpy as N

def materialize(ns):
    for w in ns.keys():
        v=ns[w]
        if not isinstance(v, ExpressionsEntry):
            continue
        v.get()

ns = env.globalns
ns.defparameter('L', central=1, sigma=0.1)

from gna.parameters.oscillation import reqparameters
reqparameters(ns)
ns['Delta'].set(N.pi*1.5)
materialize(ns)

checkpmns = ns('checkpmns')
exprs = R.PMNSExpressionsC(ns=checkpmns)
materialize(checkpmns)

ns1 = ns('weights_ae_ae')
with ns1:
    re1=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), ns=ns1)
    materialize(ns1)
# w12 = 2 sqr c12*s12*c13*c13
# w13 = 2 sqr c12*c13*s13
# w23 = 2 sqr s12*c13*s13

ns2 = ns('weights_mu_e')
with ns2:
    re2=R.OscProbPMNSExpressions(R.Neutrino.mu(), R.Neutrino.e(), ns=ns2)
    materialize(ns2)

ns3 = ns('weights_amu_ae')
with ns3:
    re3=R.OscProbPMNSExpressions(R.Neutrino.amu(), R.Neutrino.ae(), ns=ns3)
    materialize(ns3)

ns.printparameters(labels=True)

import IPython
IPython.embed()

