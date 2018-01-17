#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import configurator, pprint
from collections import OrderedDict

cfg = configurator()
cfg.detectors = [ 'AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34' ]
cfg.groups    = OrderedDict( [
                ( 'EH1', ('AD11', 'AD12') ),
                ( 'EH2', ('AD21', 'AD22') ),
                ( 'EH3', ('AD31', 'AD32', 'AD33', 'AD34') ),
                ] )

bkg = cfg('bkg')
bkg.list = [ 'acc', 'lihe', 'fastn', 'amc', 'alphan' ]

acc = cfg.bkg('acc')

pprint( cfg )
import IPython
IPython.embed()
