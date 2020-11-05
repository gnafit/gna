#!/usr/bin/env python

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
pdg_year = 2018
reqparameters_reactor(nsreac, '23', pdg_year=pdg_year)
reqparameters(nsosc, pdg_year=pdg_year)
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

def testconv_angle(ns_single, ns_double, angle):
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

    double_direct = direct_v(range_direct)
    ax.plot(range_direct, double_direct, '--', alpha=0.6, label='direct')
    ax.plot(inv_v(range_inv),  range_inv,         ':',  alpha=0.8, label='inverse')
    ax.legend()

    diff = range_direct - inv_v(double_direct)
    assert (N.fabs(diff.astype('d'))<1.e-14).all()

def testconv_dm(ns_ee, ns_23, ordering):
    alpha = ordering=='NO' and 'normal' or 'inverted'
    ns_ee['Alpha'].set(alpha)
    ns_23['Alpha'].set(alpha)

    name_ee = 'DeltaMSqEE'
    name_23 = 'DeltaMSq23'
    name_13 = 'DeltaMSq13'

    def direct(v):
        ns_ee[name_ee].set(v)
        ret = ns_ee[name_23].value(), ns_ee[name_13].value()
        return ret
    direct_v = N.frompyfunc(direct, 1, 2)

    def inv(v):
        ns_23[name_23].set(v)
        ret = ns_23[name_ee].value(), ns_23[name_13].value()
        return ret
    inv_v = N.frompyfunc(inv, 1, 2)

    dmee_in = N.linspace(1.0e-3, 3.0e-3, 200)
    dm23_out, dm13_out1 = inv_v(dmee_in)
    dmee_out, dm13_out2 = direct_v(dm23_out)

    fig = plt.figure()
    ax = plt.subplot(111,
            xlabel=r'EE: $\Delta m^2_\mathrm{ee}$',
            ylabel=r'$\Delta m^2_{ij}$',
            title=r'Test conversion for $\Delta m^2$ ({})'.format(ordering))
    ax.minorticks_on()
    ax.grid()

    ax.plot(dmee_in, dm23_out,   linestyle=(0, (3, 3)), alpha=0.6, label='23 from ee')
    ax.plot(dmee_out, dm23_out,  linestyle=(3, (3, 3)), alpha=0.6, label='ee from 23')

    ax.plot(dmee_in, dm13_out1,  linestyle=(0, (3, 3)), alpha=0.6, label='13 from ee')
    ax.plot(dmee_out, dm13_out2, linestyle=(3, (3, 3)), alpha=0.6, label='13 from 23')

    ax.legend()

    # fig = plt.figure()
    # ax = plt.subplot(111,
            # xlabel=r'EE: $\Delta m^2_\mathrm{ee}$',
            # ylabel=r'$\Delta m^2_{ij}$',
            # title=r'Test conversion for $\Delta m^2$, diff ({})'.format(ordering))
    # ax.minorticks_on()
    # ax.grid()

    # diff1 = dmee_in-dmee_out
    # diff2 = dm13_out1-dm13_out2
    # ax.plot(dmee_in, diff1,  linestyle=(0, (3, 3)), alpha=0.6, label='EE')
    # ax.plot(dmee_in, diff2,  linestyle=(0, (3, 3)), alpha=0.6, label='13')

    # ax.legend()

    assert (N.fabs(diff1.astype('d'))<1e-18).all()
    assert (N.fabs(diff2.astype('d'))<1e-18).all()

testvalues()
testconv_angle(nsosc, nsreac, 12)
testconv_angle(nsosc, nsreac, 13)
testconv_dm(nsosc, nsreac, 'NO')
testconv_dm(nsosc, nsreac, 'IO')
plt.show()

