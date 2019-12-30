#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from matplotlib import pyplot as plt
from gna.env import env, ExpressionsEntry
import numpy as N

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-c', '--pmnsc', action='store_true' )
opts = parser.parse_args()

nsosc = env.globalns('osc')
nsreac = env.globalns('reac')

from gna.parameters.oscillation import reqparameters, reqparameters_reactor
print('Initialize variables')
reqparameters_reactor(nsreac, '23')
reqparameters(nsosc)
nsreac['Delta'].set(N.pi*1.5)
nsosc['Delta'].set(N.pi*1.5)

print('Init OP ae-ae')
with nsreac:
    pmnsreac=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), ns=nsreac)
    print('Materialize expresions')
    nsreac.materializeexpressions()

with nsosc:
    pmnsosc=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), ns=nsosc)
    print('Materialize expresions')
    nsosc.materializeexpressions()
#
print('Status:')
env.globalns.printparameters(labels=True)
print()

def testvalues():
    totest = []
    for comp in (0, 12, 13, 23):
        weight = 'weight{}'.format(comp)
        v1 = nsreac[weight].value()
        v2 = nsosc[weight].value()

        totest.append(v1-v2)

    for i in range(3):
        for j in range(3):
            name = 'V{}{}'.format(i, j)
            var1 = nsreac[name].getVariable()
            var2 = nsosc[name].getVariable()

            v1=var1.value(0)
            v2=var2.value(0)
            totest.append(v1-v2)

            v1=var1.value(1)
            v2=var2.value(1)
            totest.append(v1-v2)

    totest = N.array(totest)
    assert (N.fabs(totest)<1.e-15).all()

def testconv(ns_single, ns_double, angle):
    name_single = 'SinSq'+str(angle)
    name_double = 'SinSqDouble'+str(angle)

    def direct(v):
        ns_single[name_single].set(v)
        return ns_single[name_double].value()
    direct_v = N.frompyfunc(direct, 1, 1)
    range_direct = N.linspace(0.0, 0.5, 200)

    def inv(v):
        ns_double[name_double].set(v)
        return ns_double[name_single].value()
    inv_v = N.frompyfunc(inv, 1, 1)
    range_inv = N.linspace(0.0, 1.0, 200)

    fig = plt.figure()
    ax = plt.subplot(111,
            xlabel=r'Single: $\sin^2 \theta_{%d}$'%(angle),
            ylabel=r'Double: $\sin^2 2\theta_{%d}$'%(angle),
            title=r'Test conversion for $\theta_{%d}$'%(angle))
    ax.minorticks_on()
    ax.grid()

    ax.plot(range_direct, direct_v(range_direct), '--', alpha=0.6, label='direct')
    ax.plot(inv_v(range_inv),  range_inv,         ':',  alpha=0.8, label='inverse')
    ax.legend()


testvalues()
testconv(nsosc, nsreac, 12)
testconv(nsosc, nsreac, 13)
plt.show()

