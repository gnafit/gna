#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.configurator import NestedDict, uncertain, uncertaindict
from gna.bundle import execute_bundle
from gna.env import env
from matplotlib import pyplot as P
from collections import OrderedDict

storage = env.globalns('storage')

cfg = NestedDict()
cfg.filename = 'output/sample_hists.root'
cfg.detectors = [ 'D1', 'D2', 'D3', 'D4' ]
cfg.groups=NestedDict([
        ('G1', ['D1', 'D2']),
        ('G2', ['D3']),
        ('G3', ['D4'])
        ])

bkg = cfg('bkg')
bkg.list = [ 'bkg_norm', 'bkg_rate' ]

bkg.bkg_norm = NestedDict(
        bundle = 'bkg_weighted_hist_v01',
        norm = uncertain( 1.0, 1.0, 'percent' ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            filename   = cfg.filename,
            format = 'hist_{}',
            variants = OrderedDict([
                ( 'D1', 'G1_D1' ),
                ( 'D2', 'G1_D2' ),
                ( 'D3', 'G2_D3' ),
                ( 'D4', 'G3_D4' ),
                ])
            )
        )

bkg.bkg_rate = NestedDict(
        bundle = 'bkg_weighted_hist_v01',
        rates = NestedDict(
            rates = uncertaindict(
                mode = 'absolute',
                G1 = (2.71, 0.90),
                G2 = (1.91, 0.73),
                G3 = (0.22, 0.07),
                )
            ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            filename   = cfg.filename,
            format = 'hist_{}',
            variants = cfg.groups.keys()
            )
        )

def make_sample_file( filename ):
    file = R.TFile( filename, 'recreate' )
    assert not file.IsZombie()

    it = 1

    name='hist'
    h = R.TH1D(name, name, 10, 0, 10 )
    h.SetBinContent( it, 1 ); it+=1
    file.WriteTObject( h )

    for gr, dets in cfg.groups.items():
        name = 'hist_{group}'.format( group=gr )
        h = R.TH1D(name, name, 10, 0, 10 )
        h.SetBinContent( it, 1 ); it+=1
        file.WriteTObject( h )

        for det in dets:
            name = 'hist_{group}_{det}'.format( group=gr, det=det )
            h = R.TH1D(name, name, 10, 0, 10 )
            h.SetBinContent( it, 1 ); it+=1
            file.WriteTObject( h )

    print('Generated file contents')
    file.ls()
    file.Close()

make_sample_file( cfg.filename )

for bkg in cfg.bkg.list:
    scfg = cfg.bkg[bkg]
    b = execute_bundle( cfg=scfg, namespaces=scfg.spectra.variants, storage=storage )


from gna.parameters.printer import print_parameters
print_parameters( env.globalns )

