import ROOT

_gna_libraries_loaded = False

if not _gna_libraries_loaded:
    ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
    ROOT.gDirectory.AddDirectory( False )
    ROOT.TH1.AddDirectory( False )

    ROOT.gSystem.Load('libGlobalNuAnalysis2.so')

    from gna import bindings
    bindings.setup(ROOT)

_gna_libraries_loaded = True
