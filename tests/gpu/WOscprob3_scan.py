#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check the WeightedSum transformation"""

from __future__ import print_function
import numpy as N
import matplotlib.pyplot as plt
from load import ROOT
from gna import constructors as C
from gna.env import env
import gna.parameters
import gna.parameters.oscillation
from gna import context, bindings
import ROOT
import time
from h5py import File
from printing import printl, nextlevel

ROOT.GNAObject

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-o', '--output', type=lambda fname: File(fname, 'w'), help='output file', required=True )
parser.add_argument('-s', '--show', action='store_true')
opts = parser.parse_args()

labels = [ 'comp0', 'item12', 'item13','item23' ]
weights = [ 'weight0', 'weight12', 'weight13', 'weight23' ]

from_nu = ROOT.Neutrino.ae()
to_nu = ROOT.Neutrino.ae()
clabels = [ 'P | &#8710;m12', 'P | &#8710;m13', 'P | &#8710;m23' ]

class OscProb3scanner(object):
    ntrials      = 100
    emin, emax = 1.0, 10.0
    def __init__(self, gpu, precision, ns=env.globalns):

        ndata=100
        self.ns = ns
        with context.set_context(manager=ndata, gpu=gpu, precision=precision) as self.manager:
            self.E_arr = N.linspace(self.emin, self.emax, 10, dtype=precision[0])  #array energy (МеV)
            self.ns.defparameter("L", central=52,sigma=0) #kilometre
            gna.parameters.oscillation.reqparameters(self.ns)
            pmnsexpr = C.OscProbPMNSExpressions(from_nu, to_nu, ns=self.ns)

            self.E = C.Points(self.E_arr, True, labels='Energy')

            with self.ns:
                self.oscprob = C.OscProb3(from_nu, to_nu, labels=clabels)
                self.unity = C.FillLike(1, labels='Unity')

                self.E >> (self.unity.fill, self.oscprob.comp12, self.oscprob.comp13, self.oscprob.comp23)

                self.ws = C.WeightedSum(weights, labels, labels='OscProb')
                self.unity          >> self.ws.sum.comp0
                self.oscprob.comp12 >> self.ws.sum.item12
                self.oscprob.comp13 >> self.ws.sum.item13
                self.oscprob.comp23 >> self.ws.sum.item23

                self.ns.materializeexpressions()
                self.ns.printparameters()
                pars = tuple(par.getVariable() for (name,par) in self.ns.walknames())
                self.manager.setVariables(C.stdvector(pars))

    def set_n(self, n):
        if n==self.E_arr.size:
            return

        self.E_arr = N.linspace(self.emin, self.emax, n, dtype=self.E_arr.dtype)
        self.E.set(self.E_arr, n)

        self.ntrials = int(10**(max(-4./6. * N.log10(n) + 5, 0.1)))
        return self.ntrials

    def test(self, parname='all', verbose=False):
        start_time = time.time()

        if parname=='all':
            def taint():
                self.ws.sum.taint()
                self.oscprob.comp12.taint()
                self.oscprob.comp13.taint()
                self.oscprob.comp23.taint()
        else:
            var = self.ns[parname].getVariable()
            def taint():
                var.taint()

        for x in range(self.ntrials):
            taint()

        end_time = time.time()
        fake_time = end_time - start_time
        # fake_time = 0.0

        start_time = time.time()
        for x in range(self.ntrials):
            taint()
            self.ws.sum.touch_global()

        end_time = time.time()
        elapsed_time = end_time - start_time

        unittime = (elapsed_time-fake_time)/self.ntrials

        if verbose:
            printl('Trial', parname)
            with nextlevel():
                printl('  Input size:', self.E_arr.size)
                printl('  Fake time', fake_time)
                printl('  Total time', elapsed_time)
                printl('  GNA time (%i trials)'%self.ntrials, elapsed_time-fake_time)
                printl('  GNA time per event', unittime)
            printl()

        return unittime

    def scan(self, parnames, npoints_list):
        out = N.zeros( (len(parnames), len(npoints_list)) )
        for inp, npoints in enumerate(npoints_list):
            ntr = self.set_n(npoints)
            printl('Test:', npoints, ntr, end='    ')
            for ipar, parname in enumerate(parnames):
                unittime = self.test(parname, False)
                printl(parname, unittime, end=';    ')

                out[ipar, inp] = unittime
            print()

        return out

parnames = ['all', 'DeltaMSqEE', 'SinSq13']
npoints_list = N.concatenate( [
    [1],
    [10],
    [100,     200,     500],
    [1000,    2000,    5000],
    [10000,   20000,   50000],
    [100000,  200000,  500000],
    [1000000, 2000000, 5000000],
    [10000000],
    ] )

opts.output.create_dataset('size', data=npoints_list)
for device in ('cpu', 'gpu'):
    dgroup = opts.output.create_group(device)
    for precision in ('double', 'float'):
        group = dgroup.create_group(precision)
        printl(device, precision)
        with nextlevel():
            ns = env.globalns(device)(precision)
            scanner = OscProb3scanner(gpu=device=='gpu', precision=precision, ns=ns)
            res = scanner.scan(parnames, npoints_list)

            for i, parname in enumerate(parnames):
                pardata = group.create_dataset(parname, data=res[i])

            group.create_dataset('enu', data=scanner.E.points.points.data())
            group.create_dataset('psur', data=scanner.ws.sum.sum.data())

        print()
        print()
print('Write output file', opts.output.filename)
opts.output.close()

if opts.show:
    from gna.bindings import common
    fig = plt.figure()
    ax = plt.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_title(r'$\overline{\nu}_e$ survival probability at 52 km')
    ax.set_xlabel(r'$E_{\nu}$, MeV')
    ax.set_ylabel(u'$P_{ee}$')
    if scanner.E_arr.size<101:
        scanner.ws.sum.sum.plot_vs(scanner.E.single(), 'o-', markerfacecolor='none')
    else:
        scanner.ws.sum.sum.plot_vs(scanner.E.single(), '-')

plt.show()

