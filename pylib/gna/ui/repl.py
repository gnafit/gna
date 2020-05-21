# -*- coding: utf-8 -*-
"""Embed IPython repl on module execution"""

from __future__ import absolute_import
from gna.ui import basecmd
import numpy as np
import ROOT
from matplotlib import pyplot as plt

class cmd(basecmd):
    def run(self):
        import IPython
        IPython.embed()
