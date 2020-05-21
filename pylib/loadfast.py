# -*- coding: utf-8 -*-
"""Loader module like `load`
doesn't initialize bindings"""
from __future__ import absolute_import
import ROOT
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
ROOT.gDirectory.AddDirectory( False )
ROOT.TH1.AddDirectory( False )
ROOT.gSystem.Load('libGlobalNuAnalysis2.so')
ROOT.PyConfig.IgnoreCommandLineOptions = True
