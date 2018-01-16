#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gna.grouping import *
from collections import OrderedDict

dets = OrderedDict( {
    '11' : 1, '12' : 2,
    '21' : 3, '22' : 8,
    '31' : 4, '32' : 5,
    '33' : 6, '33' : 7
    } )

sites = OrderedDict( {
    'EH1' : 'A',
    'EH2' : 'B',
    'EH3' : 'C'
    } )

groups = OrderedDict( {
    'EH1' : ('11', '12'),
    'EH2' : ('21', '22'),
    'EH3' : ('31', '32', '33', '34'),
    } )

gd = GroupedDict( sites, groups=groups )

import IPython
IPython.embed()
