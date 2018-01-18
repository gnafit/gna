#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.configurator import NestedDict
from gna.bundle import execute_bundle
from gna.env import env

storage = env.globalns('storage')

cfg = NestedDict()
cfg.filename = 'output/sample_hists.root'
cfg.detectors = [ 'D1', 'D2', 'D3', 'D4' ]
cfg.groups=NestedDict(
        G1 = [ 'D1', 'D2' ],
        G2 = [ 'D3' ],
        G3 = [ 'D4' ]
        )

cfg.spectra = [ 'spectra1', 'spectra2', 'spectra3' ]
cfg.spectra1 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        # format = 'hist_{group}_{det}',
        format = 'hist_{}',
        variants = cfg.detectors
        )
cfg.spectra2 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        format = 'hist_{}',
        variants = cfg.groups.keys()
        )
cfg.spectra3 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        format = 'hist',
        )

def make_sample_file( filename ):
    file = R.TFile( filename, 'recreate' )
    assert not file.IsZombie()

    name='hist'
    h = R.TH1D(name, name, 10, 0, 10 )
    file.WriteTObject( h )
    for gr, dets in cfg.groups.items():
        name = 'hist_{group}'.format( group=gr )
        h = R.TH1D(name, name, 10, 0, 10 )
        file.WriteTObject( h )

        for det in dets:
            name = 'hist_{det}'.format( group=gr, det=det )
            h = R.TH1D(name, name, 10, 0, 10 )
            file.WriteTObject( h )

    print('Generated file contents')
    file.ls()
    file.Close()

make_sample_file( cfg.filename )

for spname in cfg.spectra:
    scfg = cfg[spname]
    b = execute_bundle( cfg=scfg, namespaces=cfg.detectors, storage=storage )

    import IPython
    IPython.embed()

    break
