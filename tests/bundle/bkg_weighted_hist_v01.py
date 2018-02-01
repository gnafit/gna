#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundle
from gna.env import env, findname
from matplotlib import pyplot as P
from mpl_tools.helpers import plot_hist, plot_bar
from collections import OrderedDict

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument( '-s', '--set', nargs=2, action='append', help='set parameter' )
opts = parser.parse_args()

storage = env.globalns('storage')

cfg = NestedDict()
cfg.filename = 'output/sample_hists.root'
cfg.detectors = [ 'D1', 'D2', 'D3', 'D4' ]
cfg.groups=NestedDict(
        exp  = { '': cfg.detectors },
        det  = { d: (d,) for d in cfg.detectors },
        site = NestedDict([
            ('G1', ['D1', 'D2']),
            ('G2', ['D3']),
            ('G3', ['D4'])
            ])
        )

bkg = cfg('bkg')
bkg.list = [ 'bkg1', 'bkg2' ]

bkg.bkg1 = NestedDict(
        bundle   = 'bkg_weighted_hist_v01',
        formula  = [ '{det}.bkg1_norm', '{det}.bkg1_rate', '{det}.livetime' ],
        groups   = cfg.groups,
        variants = cfg.detectors,

        bkg1_norm = uncertaindict([ (det, (1.0, 1.0, 'percent')) for det in cfg.detectors ]),

        bkg1_rate = uncertaindict(
            mode = 'fixed',
            D1 = 8,
            D2 = 7,
            D3 = 4,
            D4 = 3,
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
        formula = [ '{site}.bkg2_rate', '{det}.livetime' ],
        groups = cfg.groups,
        variants = cfg.detectors,

        bkg2_rate = uncertaindict(
            mode = 'absolute',
            G1 = (2.71, 0.90),
            G2 = (1.91, 0.73),
            G3 = (0.22, 0.07),
            ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            filename   = cfg.filename,
            format = 'hist_{}',
            variants = cfg.groups['site'].keys(),
            normalize = True,
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

bundles=()
for bkg in cfg.bkg.list:
    scfg = cfg.bkg[bkg]
    b = execute_bundle( cfg=scfg, common_namespace=ns, namespaces=scfg.spectra.variants, storage=storage )
    bundles+=b,

from gna.parameters.printer import print_parameters
print_parameters( env.globalns )

if opts.set:
    for name, value in opts.set:
        value = float(value)
        print( 'Set', name, value )
        var = findname( ns.pathto(name), ns )
        var.set( value )

    from gna.parameters.printer import print_parameters
    print_parameters( env.globalns )

for bundle in bundles:
    fig = P.figure()
    ax = P.subplot( 111 )
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel( 'x axis' )
    ax.set_ylabel( 'entries' )
    ax.set_title( bundle.cfg.name )

    for name, ts in bundle.transformations.items():
        output = ts.sum.single()
        if bundle.cfg.name=='bkg2':
            group = bundle.groups.get_group(name, 'site')
            pack = (group.index(name), len(group))
        else:
            pack = None
        plot_bar( output.datatype().edges, output.data(), label=name, pack=pack )

    ax.legend( loc='upper right' )

P.show()


