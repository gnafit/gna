#!/usr/bin/env python

from load import ROOT as R
from gna.configurator import NestedDict
from gna.bundle import execute_bundles
from gna.env import env
from matplotlib import pyplot as P

cfg = NestedDict()
cfg.filename = 'output/sample_hists.root'
cfg.detectors = [ 'D1', 'D2', 'D3', 'D4' ]
cfg.groups=NestedDict([
        ('G1', ['D1', 'D2']),
        ('G2', ['D3']),
        ('G3', ['D4'])
        ])

cfg.spectra = [ 'spectra1', 'spectra2', 'spectra3' ]
cfg.spectra1 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        format = 'hist',
        )
cfg.spectra2 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        format = 'hist_{self}',
        variants = cfg.groups.keys()
        )
cfg.spectra3 = NestedDict(
        bundle = 'root_histograms_v01',
        filename = cfg.filename,
        format = 'hist_{self}',
        variants = dict([
            ( 'D1', 'G1_D1' ),
            ( 'D2', 'G1_D2' ),
            ( 'D3', 'G2_D3' ),
            ( 'D4', 'G3_D4' ),
            ])
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

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'bin' )
ax.set_ylabel( 'height' )
ax.set_title( 'Histogram' )
for spname in cfg.spectra:
    scfg = cfg[spname]
    b, = execute_bundles(cfg=scfg)

    for output, ns in zip(b.outputs.values(), b.namespaces):
        data = output.data()
        ax.bar(range(len(data)), data, label=ns.name)

ax.legend()

print( 'Walk namespaces:' )
for ns in env.globalns.walknstree():
    print( '   ', ns )

P.show()
