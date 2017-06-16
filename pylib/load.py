import ROOT
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
ROOT.gDirectory.AddDirectory( False )
ROOT.TH1.AddDirectory( False )

ROOT.gSystem.Load('libGlobalNuAnalysis2.so')
