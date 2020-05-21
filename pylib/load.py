# -*- coding: utf-8 -*-
#
# Make sure python libs are loaded before ROOT
# it seems that recent ROOT versions do have precompiled scipy
# which may be in conflict with system scipy version (something with QHUL library)
#
from __future__ import absolute_import
import numpy
import matplotlib
import scipy.stats

import ROOT

ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
ROOT.gDirectory.AddDirectory( False )
ROOT.TH1.AddDirectory( False )
ROOT.PyConfig.IgnoreCommandLineOptions = True

ROOT.gSystem.Load('libGlobalNuAnalysis2')

from gna import bindings
bindings.setup(ROOT)
