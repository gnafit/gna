#
# Make sure python libs are loaded before ROOT
#
import numpy
import matplotlib
import scipy.stats

import ROOT

ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
ROOT.gDirectory.AddDirectory( False )
ROOT.TH1.AddDirectory( False )

ROOT.gSystem.Load('libGlobalNuAnalysis2.so')

from gna import bindings
bindings.setup(ROOT)
