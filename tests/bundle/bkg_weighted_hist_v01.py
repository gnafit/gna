#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundle
from gna.env import env, findname
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar
from collections import OrderedDict

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--set', nargs=2, action='append', help='set parameter' )
parser.add_argument( '-p', '--pack', action='store_true', help='pack same site histograms' )
opts = parser.parse_args()

storage = env.globalns('storage')

cfg = NestedDict()
cfg.filename = 'output/sample_hists.root'
cfg.detectors = [ 'D1', 'D2', 'D3', 'D4' ]
cfg.groups=NestedDict(
        exp  = { 'testexp': cfg.detectors },
        det  = { d: (d,) for d in cfg.detectors },
        site = NestedDict([
            ('G1', ['D1', 'D2']),
            ('G2', ['D3']),
            ('G3', ['D4'])
            ])
        )

bkg = cfg('bkg')
bkg.bundle = 'bundlesum_v01'
bkg.list = [ 'bkg1', 'bkg2', 'bkgw' ] #, 'bkg_fn'
bkg.observable = 'bkg_total'

bkg.bkg1 = NestedDict(
        bundle   = 'bkg_weighted_hist_v01',
        formula  = [ '{det}.bkg1_num', ('bkg1_norm.{det}', '{det}.bkg1_rate', '{det}.livetime') ],
        groups   = cfg.groups,
        variants = cfg.detectors,

        bkg1_norm = uncertaindict([
            (det, (1.0, 1.0, 'percent')) \
              for det in cfg.detectors
            ]),

        bkg1_rate = uncertaindict(
              [ ('D1', 8),
                ('D2', 7),
                ('D3', 4),
                ('D4', 3) ],
                mode = 'fixed',
            ),

        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            filename   = cfg.filename,
            format = 'hist_{}',
            variants = OrderedDict([
                ( 'D1', 'G1_D1' ),
                ( 'D2', 'G1_D2' ),
                ( 'D3', 'G2_D3' ),
                ( 'D4', 'G3_D4' ),
                ]),
            normalize = True,
            )
        )

bkg.bkg2 = NestedDict(
        bundle = 'bkg_weighted_hist_v01',
        formula = '{det}.bkg2_num=bkg2_rate.{site}*{det}.livetime' ,
        groups = cfg.groups,
        variants = cfg.detectors,

        bkg2_rate = uncertaindict(
           [('G1',  (2.71, 0.90)),
            ('G2',  (1.91, 0.73)),
            ('G3',  (0.22, 0.07))],
            mode = 'absolute',
            ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            filename   = cfg.filename,
            format = 'hist_{}',
            variants = cfg.groups['site'].keys(),
            normalize = True,
            )
        )

bkg.bkgw = NestedDict(
        bundle = 'bkg_weighted_hist_v01',
        formula = [ '{det}.bkgw', ('bkgw.{site}', '{det}.livetime') ],
        groups = cfg.groups,
        variants = cfg.detectors,

        bkgw = uncertaindict(
           [('G1', (1.0, 0.3)),
            ('G2', (3.0, 0.2)),
            ('G3', (2.0, 0.1))],
            mode = 'absolute',
            ),
        spectra = NestedDict(
            bundle = 'hist_mixture_v01',

            fractions = uncertaindict(
                li = ( 0.90, 0.05, 'relative' )
                ),
            spectra = NestedDict([
                ('li', NestedDict(
                    bundle = 'root_histograms_v01',
                    filename   = cfg.filename,
                    format = 'hist_G1_D1',
                    normalize = True,
                    )),
                ('he', NestedDict(
                    bundle = 'root_histograms_v01',
                    filename   = cfg.filename,
                    format = 'hist_G2_D3',
                    normalize = True,
                    )),
                ])
            )
        )

bkg.bkg_fn = NestedDict(
        bundle = 'bkg_weighted_hist_v01',
        formula = [ '{det}.bkg_fn_num', ('bkg_fn_rate.{site}', '{det}.livetime') ],
        groups = cfg.groups,
        variants = cfg.detectors,

        bkg_fn_rate = uncertaindict(
           [('G1', (1.0, 0.3)),
            ('G2', (3.0, 0.2)),
            ('G3', (2.0, 0.1))],
            mode = 'absolute',
            ),
        spectra = NestedDict(
            bundle='dayabay_fastn_v01',
            formula='fastn_shape.{site}',
            groups=cfg.groups,
            normalize=(0.7, 12.0),
            bins =N.linspace(0.0, 12.0, 241),
            order=2,
            pars=uncertaindict(
               [ ('G1', (70.00, 0.1)),
                 ('G2', (60.00, 0.05)),
                 ('G3', (50.00, 0.2)) ],
                mode='relative',
                ),
            )
        )

def make_sample_file( filename ):
    file = R.TFile( filename, 'recreate' )
    assert not file.IsZombie()

    name='hist'
    h = R.TH1D(name, name, 10, 0, 10 )
    h.SetBinContent( 1, 1 )
    file.WriteTObject( h )

    it_site, it_det = 1, 1
    for gr, dets in cfg.groups['site'].items():
        name = 'hist_{group}'.format( group=gr )
        h = R.TH1D(name, name, 10, 0, 10 )
        h.SetBinContent( it_site, 1 ); it_site+=1
        file.WriteTObject( h )

        for det in dets:
            name = 'hist_{group}_{det}'.format( group=gr, det=det )
            h = R.TH1D(name, name, 10, 0, 10 )
            h.SetBinContent( it_det, 1 ); it_det+=1
            file.WriteTObject( h )

    print('Generated file contents')
    file.ls()
    file.Close()

make_sample_file( cfg.filename )

ns = env.globalns('testexp')
for det in cfg.detectors:
    detns = ns(det).reqparameter('livetime', central=10, sigma=0.1, fixed=True)

b, = execute_bundle( cfg=bkg, common_namespace=ns )
bfn, = execute_bundle( cfg=bkg.bkg_fn, common_namespace=ns )

print( 'Parameters:' )
env.globalns.printparameters()

print( 'Observables:' )
env.globalns.printobservables()

if opts.set:
    for name, value in opts.set:
        value = float(value)
        print( 'Set', name, value )
        var = findname( ns.pathto(name), ns )
        var.set( value )

    from gna.parameters.printer import print_parameters
    print_parameters( env.globalns )

for bundle in b.bundles.values()+[b, bfn]:
    bkgname=bundle.cfg.get('name', 'total')

    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'x axis' )
    ax.set_ylabel( 'entries' )
    ax.set_title(bkgname)

    for i, (name, output) in enumerate(bundle.outputs.items()):
        pack=None
        if opts.pack:
            if bkgname=='bkg2':
                group = bundle.groups.get_group(name, 'site')
                pack = (group.index(name), len(group))
            elif bkgname=='bkgw':
                pack = (i, len(bundle.outputs))

        if bkgname=='bkg_fn':
            plot_hist( output.datatype().edges, output.data(), label=name )
        else:
            plot_bar( output.datatype().edges, output.data(), label=name, pack=pack, alpha=0.8 )

    ax.legend( loc='upper right' )

P.show()


