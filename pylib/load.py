#
# Make sure python libs are loaded before ROOT
# it seems that recent ROOT versions do have precompiled scipy
# which may be in conflict with system scipy version (something with QHUL library)
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
