#
# Make sure python libs are loaded before ROOT
# it seems that recent ROOT versions do have precompiled scipy
# which may be in conflict with system scipy version (something with QHUL library)
#
import numpy
import matplotlib
import scipy.stats

from os import environ
if not environ.get('DISPLAY'):
    matplotlib.use('Agg')

import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
ROOT.gDirectory.AddDirectory( False )
ROOT.TH1.AddDirectory( False )

ROOT.gSystem.Load('libGlobalNuAnalysis2')

from gna import bindings
bindings.setup(ROOT)
