#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import configurator
from collections import OrderedDict

cfg = configurator()
cfg.detectors = [ '11', '12', '21', '22', '31', '32', '33', '34' ]
cfg.groups    = OrderedDict( [
                ( 'EH1', ('11', '12') ),
                ( 'EH2', ('21', '22') ),
                ( 'EH3', ('31', '32', '33', '34') ),
                ] )

import IPython
IPython.embed()
