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

# Let GNA parse the command line options
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.PyConfig.Shutdown = False
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
# Disable automatic addition of the objects to directories
ROOT.gDirectory.AddDirectory(False)
ROOT.TH1.AddDirectory(False)

ROOT.gSystem.Load('libGlobalNuAnalysis2')

from gna import bindings
bindings.setup(ROOT)
